from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openai import OpenAI
import tiktoken
import json
import re
import fitz
import os
import hmac
import hashlib
from supabase_client import fetch_supabase_db, fetch_supabase_cat_db, upsert_category
from pydantic import BaseModel
import requests
import asyncio
import logging
import io
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv # for access .env folder
load_dotenv()


from supabase import create_client  # add this import

SUPABASE_URL = "https://tmpsadthcxvqtdbtslgq.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_PRIVATE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Force logs to stdout and set formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("app")


# ✅ NEW FUNCTION: Update statement status in Supabase
def update_statement_status(import_id: str, status: str, error: str = None, processor_job_id: str = None):
    """
    Update statement_imports table status column in Supabase.
    
    Args:
        import_id: UUID of the statement_imports record (from React Native app)
        status: 'uploaded' | 'processing' | 'completed' | 'failed'
        error: Error message if status is 'failed' (optional)
        processor_job_id: Optional job ID from your processing queue
    """
    update_data = {
        "status": status
    }
    
    # Add processed_at timestamp for completed/failed statuses
    if status in ["completed", "failed"]:
        update_data["processed_at"] = datetime.utcnow().isoformat()
    
    # Add error message if status is failed
    if status == "failed" and error:
        update_data["error"] = error
    
    # Add processor job ID if provided
    if processor_job_id:
        update_data["processor_job_id"] = processor_job_id
    
    try:
        result = supabase.table("statement_imports").update(update_data).eq("id", import_id).execute()
        logger.info(f"✅ Updated statement {import_id} to status: {status}")
        return result
    except Exception as e:
        logger.error(f"❌ Error updating statement status: {e}")
        raise


# -------------------- Helpers --------------------

def safe_parse_json(raw):
    """
    Try to parse JSON, fix minor truncation issues if possible.
    ✅ IMPROVED: Handles extra closing brackets like ]] and other malformed JSON
    """
    raw = raw.strip()
    # Remove or ``` markdown if present
    raw = re.sub(r"^", "", raw)
    raw = re.sub(r"```$", "", raw)
    raw = raw.strip()

    # ✅ FIX: Remove extra closing brackets at the end
    # Handle cases like: [...]] or [...]]]
    while raw.endswith(']') and raw.count(']') > raw.count('['):
        raw = raw.rstrip(']').rstrip()
    
    # ✅ FIX: Remove extra closing braces at the end
    # Handle cases like: [...}} or [...}}}
    while raw.endswith('}') and raw.count('}') > raw.count('{'):
        raw = raw.rstrip('}').rstrip()

    # Try normal parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to fix common issues
        # If it starts with [ but doesn't end with ], try adding ]
        if raw.startswith("[") and not raw.endswith("]"):
            raw += "]"
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON from the string if it's embedded
        # Look for JSON array pattern
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            try:
                extracted = json_match.group(0)
                # Clean up extracted JSON
                while extracted.endswith(']') and extracted.count(']') > extracted.count('['):
                    extracted = extracted.rstrip(']').rstrip()
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Failed to parse JSON. Raw output length: {len(raw)}")
        logger.error(f"First 500 chars: {raw[:500]}")
        logger.error(f"Last 500 chars: {raw[-500:]}")
        return None


# --- Helper to chunk list ---
def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]


# --- Count tokens function ---
def count_tokens(prompt, model="gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


# Core processing functions 

def categorize_transactions_batch(client, df, amount_threshold=100, batch_size=20, model="gpt-4o-mini", person_name='Abhishek', mobile_numbers='7206527787'):
    """
    Categorize transactions using LLM in batches with confidence scores.
    Handles 'Narration', 'Credit Amount', 'Debit Amount', and 'Date' columns.
    Returns df with new 'Amount', 'Category', 'Confidence' columns.
    """
    results = []

    # Unified Amount column
    df["Amount"] = df["Credit Amount"].fillna(0) - df["Debit Amount"].fillna(0)
    df.rename(columns={"Narration": "Description"}, inplace=True)

    # Filter rows for LLM
    df_to_process = df[df["Amount"].abs() >= amount_threshold].copy()
    
    # ✅ FIX: Check if there are any transactions to process
    if df_to_process.empty:
        logger.warning(f"No transactions found above threshold {amount_threshold}")
        return pd.DataFrame()

    for batch in chunker(list(df_to_process.itertuples(index=False)), batch_size):
        # Prepare transactions text including Date
        transactions_text = "\n".join(
            [f"- Description: {row.Description}, Amount: {row.Amount}, Date: {row.Date}" for row in batch]
        )

        # Prompt with confidence and date
        prompt = f"""
        You are an expert accounting assistant. Categorize the following bank transactions using Description, Amount, and Date.
        
        Rules for categorization:
        1. Categories include (but are not limited to): 
           Food, Travel, Office Supplies, Client Expense, Dividend, Investment, Investment Withdrawal, 
           Profit from Investment, Interest, Tax, Regular recurring Salary, UPI Transfer, Shopping, 
           Entertainment, Rent, Self transfer, Mobile Recharge, Miscellaneous.
        
        2. Specific clarifications:
           - If Amount > 0 (deposit/credit), it CANNOT be categorized as "Investment".
             • If it is a return/profit/redemption from investment, use "Profit from Investment" or "Investment Withdrawal".
           - If Description contains "ACH D- INDIAN CLEARING CORP" or similar clearing corp, categorize as "Investment".
           - If Description contains "ZEPTO", "ZOMATO", "SWIGGY", or other food merchants, categorize as "Food".
           - For UPI transactions:
             • If it matches known merchants, map accordingly.
             • If it is clearly P2P (names/emails/phone numbers), categorize as "UPI Transfer".
             • If the transaction description or UPI ID contains the **account holder name or mobile number(s)** provided to you (see *Person data* below), treat it as a "Self transfer" when sender and receiver are the same person.
           - Salary credits should be mapped to "Regular recurring Salary".
        
        3. **Consistency rule (NEW)**:
           - If multiple transactions in this batch share the same normalized merchant name (normalize by removing numeric IDs, UTR strings, `CR/DR` tokens, and punctuation), ensure they are assigned the **same Category** across the batch. If the model is unsure, choose the category that appears most frequent among those similar transactions (majority). If a tie, choose the category with higher confidence.
        
        4. **Person data (INPUT; NEW)**:
           - Person name (from function): `{person_name}`
           - Mobile numbers (from function): `{mobile_numbers}`
           - Use these to identify "Self transfer" — i.e., if the narration contains the same person name or any of the mobile numbers, prefer "Self transfer" when the flow looks like an internal move.
        
        5. **Confidence labels (NEW)**:
           - Instead of numeric confidence, return one of: `"very high"`, `"high"`, `"medium"`, `"low"`.
           - Use `"very high"` when a deterministic rule or exact merchant match applies (e.g., merchant in known list).
           - Use `"high"` for strong semantic matches or clear patterns.
           - Use `"medium"` for plausible guesses.
           - Use `"low"` if you are unsure or the narration is ambiguous.

        Output format:
        Respond ONLY in valid JSON format as a list of objects with keys:
        Description, Amount, Date, Category, Confidence, Reason
        
        Notes:
        - Reason must be 1–3 words only, explaining why the Category was chosen.
          Example: "Zepto", "salary credit Paytm", "p2p Vijay", "investment corp ACH", "investment corp Zerodha" etc.
        - IMPORTANT: Return ONLY the JSON array, no extra brackets or closing characters.
        
        Transactions:
        {transactions_text}
        """

        # Optional: print token length
        prompt_length = count_tokens(prompt, model=model)
        print("🔹 Prompt token length:", prompt_length)
        logger.info(f"Prompt length for main classifier call: {prompt_length}")

        # LLM call
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
        )

        # Parse JSON output
        logger.info(f"pdf to csv response: {response}")
        raw_output = response.choices[0].message.content
        logger.info("LLM response generated from main classifier")
        batch_results = safe_parse_json(raw_output)
        if batch_results:
            results.extend(batch_results)
            logger.info(f"Successfully parsed {len(batch_results)} transactions from batch")
        else:
            print("⚠️ Failed to parse JSON for batch. Raw output:", raw_output[:200])
            logger.warning("Failed to parse JSON for the batch from main classifier")
            logger.warning(f"Raw output (first 500 chars): {raw_output[:500]}")

    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        logger.info(f"Successfully categorized {len(results_df)} transactions")
    else:
        logger.warning("No transactions were successfully categorized")
        results_df = pd.DataFrame()

    return results_df


def extract_text(doc):
    all_lines = []
    for page in doc:
        text = page.get_text("text")
        lines = text.split('\n')
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        all_lines.extend(lines)
    extracted_text = "\n".join(all_lines)
    return extracted_text


def download_file_from_s3(presigned_url):
    response = requests.get(presigned_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF: {response.status_code}")

    return response


def pdf_to_csv(file_response, client, model):
    pdf_bytes = file_response.content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = extract_text(doc)

    # Construct the prompt
    prompt = f"""
        You are an AI assistant specialized in parsing bank statements.
        Your task is to extract all transactions into strictly the following CSV columns:
        Date,Narration,Debit Amount,Credit Amount

        Guidelines:
        - Output ONLY raw CSV, NO explanations or Markdown.
        - The FIRST LINE must be the header: Date,Narration,Debit Amount,Credit Amount
        - Each row should ONLY be: Date,Narration,Debit Amount,Credit Amount
        - Leave fields blank if data not available, but keep all four columns.

        Here is the extracted text:
        {extracted_text}
    """
    prompt_length = count_tokens(prompt, model="gpt-4o-mini")
    print("🔹 Prompt token length:", prompt_length)
    logger.info("Prompt length for converting pdf to csv: {prompt_length}")

    # Make the OpenAI API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000
    )

    # Extract the CSV from the response
    csv_output = response.choices[0].message.content
    logger.info("LLM response generated for pdf to csv")
    # Save to CSV
    with open("bank_statement_parsed.csv", "w", encoding="utf-8") as f:
        f.write(csv_output)
    df = pd.read_csv("bank_statement_parsed.csv")
    logger.info("PDF to csv file written and read to return df")

    return df


def csv_col_identify(cols, client, model):
    """
    Identify the columns for various bank statements.
    Handles 'Narration', 'Credit Amount', 'Debit Amount', and 'Date' columns.
    Returns a dictionary mapping each logical name to the actual column name.
    """
    prompt = f"""
    You are a bank statement structure identifier.
    Identify which columns from the list provided correspond to:
      - 'Narration' (transaction description)
      - 'Credit Amount' (incoming money)
      - 'Debit Amount' (outgoing money)
      - 'Date' (transaction date)

    Rules:
    - If you are not sure, make the best guess based on column name semantics (e.g., "cr", "dr", "txn_date", "details", "withdraw", "deposit").
    - Return ONLY valid JSON with keys exactly as:
        {{<column_name>: "Narration", <column_name> : "Credit Amount", <column_name> : "Debit Amount", <column_name> : "Date"}}

    Columns:
    {cols}
    """

    # LLM call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    # Extract text content
    raw_output = response.choices[0].message.content.strip()

    # Remove markdown if present (liken ... ```)
    raw_output = raw_output.replace("", "").replace("```", "").strip()

    # Parse into dictionary safely
    try:
        mapping = json.loads(raw_output)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON from LLM output. Raw output:")
        print(raw_output)
        mapping = {}

    return mapping



# ✅ UPDATED: df_to_event_list - Now includes file_id parameter for junction table
def df_to_event_list(df, client_id, accountant_id, file_id=None):
    """
    Convert DataFrame to event list for webhook.
    Now includes file_id to link transactions to statement file via junction table.
    """
    # ✅ FIX: Check if DataFrame is empty or None
    if df is None or df.empty:
        logger.warning("DataFrame is empty or None, returning empty event list")
        return []
    
    # ✅ FIX: Check if required columns exist
    required_cols = ['Category', 'Confidence', 'Reason', 'Description', 'Amount', 'Date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        error_msg = f"Missing required columns in DataFrame: {missing_cols}. Available columns: {list(df.columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Select required columns and convert to list of dicts
    event_list = df[required_cols].to_dict(orient='records')

    cat_id_map = fetch_supabase_cat_db()
    name_to_id_map = {item['name']: item['id'] for item in cat_id_map}
    
    # Add additional info to each event object
    for event in event_list:
        event['client_id'] = client_id
        event['accountant_id'] = accountant_id
        event['file_id'] = file_id  # ✅ Keep file_id for junction table linking
        if event['Category'] not in name_to_id_map:
            res = upsert_category(event['Category'])
            name_to_id_map[event['Category']] = res[0]['id']

        event['category_id'] = name_to_id_map[event['Category']]
        event['confidence'] = event['Confidence']
        event['reason'] = event['Reason']
        event['tx_narration'] = event['Description']
        event['tx_amount'] = event['Amount']
        event['tx_timestamp'] = event['Date']
        del event['Category']
        del event['Confidence']
        del event['Reason']
        del event['Description']
        del event['Amount']
        del event['Date']

    return event_list


#Webhook HMAC helpers & invoker

def generate_hmac_sha256_signature(secret_key, message):
    # Convert both secret and message to bytes
    key_bytes = secret_key.encode('utf-8')
    message_bytes = message.encode('utf-8')

    # Create HMAC SHA256 signature
    signature = hmac.new(key_bytes, message_bytes, hashlib.sha256).hexdigest()
    return signature


def invoke_webhook(event_list):
    var_json = {"events": event_list}

    logger.info("Starting invoke webhook")
    raw_json_string = json.dumps(var_json)

    # Read target webhook URL from environment so you can change per-deploy
    webhook_url = "https://personal-finance-manager-python.onrender.com/transactions/webhook"

    signature = generate_hmac_sha256_signature(os.getenv("WEBHOOK_SIGNATURE_KEY", ""), raw_json_string)
    headers = {"Content-Type": "application/json", "x-signature": signature}

    logger.info("Invoking webhook event to %s", webhook_url)
    try:
        # ✅ FIX: Increase timeout to 120 seconds (Render.com can be slow on cold starts)
        response = requests.post(webhook_url, data=raw_json_string, headers=headers, timeout=120)
        if response.status_code == 200:
            logger.info("Webhook success")
            print("Webhook successfully invoked.")
        else:
            logger.error(f"Webhook invocation failed with status code {response.status_code} - Response: {response.text}")
            print(f"Webhook invocation failed: {response.status_code} - {response.text}")
        return response  # ✅ Always return response object
    except requests.exceptions.Timeout:
        # ✅ FIX: Handle timeout separately - webhook might still process transactions
        logger.warning("Webhook call timed out after 120 seconds. Transactions may still be processed.")
        logger.warning("This is common on Render.com free tier due to cold starts.")
        # Return a mock response object to indicate timeout
        class TimeoutResponse:
            status_code = 408  # Request Timeout
            text = "Webhook call timed out (may still be processing)"
        return TimeoutResponse()
    except Exception as e:
        logger.error("Exception while invoking webhook: %s", str(e))
        return None  # ✅ Return None on other exceptions


# -------------------- FastAPI app & routes --------------------

app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/hello")
def say_hello():
    return {"message": "Hello from the named API!"}


# UPDATED: ClassifierRequest - Removed file_id field

class ClassifierRequest(BaseModel):
    import_id: Optional[str] = None  # import_id from statement_imports table
    client_id: str
    signed_url: str
    accountant_id: Optional[str] = None
    # Keep backward compatibility with old fields
    user_id: Optional[str] = None  # Also accept user_id (same as client_id)
    file_url: Optional[str] = None  # Also accept file_url (alternative to signed_url)


@app.post("/classifier")
async def classifier_api(request: ClassifierRequest):
    """
    UPDATED: Now handles status updates throughout the processing lifecycle
    """
    import_id = request.import_id
    client_id = request.client_id or request.user_id  # Support both field names
    
    # NEW: If import_id is provided, update status to 'processing' immediately
    if import_id:
        try:
            update_statement_status(import_id, 'processing', processor_job_id=f"job_{import_id[:8]}")
            logger.info(f"Updated statement {import_id} to 'processing' status")
        except Exception as e:
            logger.error(f"⚠️ Failed to update status to 'processing': {e}")
            # Continue processing even if status update fails
    
    file_list = []
    
    # NEW: Support both signed_url and file_url
    file_url = request.signed_url or request.file_url
    
    try:
        # download files using presigned urls
        file_list.append(download_file_from_s3(file_url))
        
        # fetch client info from supabase db
        client_info = fetch_supabase_db(client_id)
        
        if client_info == -1:
            error_msg = f"Client not found: {client_id}"
            logger.error(error_msg)
            # NEW: Update status to 'failed' if client not found
            if import_id:
                update_statement_status(import_id, 'failed', error=error_msg)
            return {"status": "error", "message": error_msg}
        
        # call classifier_main - async call
        await classifier_main(
            file_list, 
            client_info['first_name'], 
            client_info['phone_number'], 
            client_id, 
            request.accountant_id,
            import_id  # Pass import_id to classifier_main
        )
        
        # NEW: Status will be updated to 'completed' or 'failed' in classifier_main
        return {"status": "completed"}
        
    except Exception as e:
        error_msg = f"Error processing statement: {str(e)}"
        logger.error(error_msg)
        # NEW: Update status to 'failed' on any exception
        if import_id:
            try:
                update_statement_status(import_id, 'failed', error=error_msg)
            except Exception as status_error:
                logger.error(f"Failed to update status to 'failed': {status_error}")
        return {"status": "error", "message": error_msg}


# UPDATED: classifier_main - Now passes file_id to df_to_event_list

async def classifier_main(file_list, name, mob_no, client_id, accountant_id, import_id=None):
    """
    UPDATED: Now accepts import_id and updates status to 'completed' or 'failed'
    Note: import_id is used as file_id for junction table linking
    """
    res_final = pd.DataFrame()
    
    # Convert import_id to file_id for clarity (easier to understand)
    file_id = import_id
    
    try:
        ## deepseek
        client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("OPENAI_API_KEY"),  # Deepseek free chat
        )
        model = "deepseek-chat"

        for file in file_list:
            try:
                if "pdf" in file.headers.get("Content-Type", ""):
                    df = pdf_to_csv(file, client, model)
                    res = categorize_transactions_batch(
                        client, df, amount_threshold=150, batch_size=50, 
                        model=model, person_name=name, mobile_numbers=mob_no
                    )
                else:
                    df = pd.read_csv(io.StringIO(file.content.decode('utf-8')))
                    ## columns intent
                    map = csv_col_identify(df.columns, client, model)
                    logger.info(f"columns from the csv files: {map}")
                    df = df.rename(columns=map)
                    # Step 2: Keep only columns present in mapping
                    df = df[[col for col in map.values() if col in df.columns]]
                    res = categorize_transactions_batch(
                        client, df, amount_threshold=150, batch_size=50, 
                        model=model, person_name=name, mobile_numbers=mob_no
                    )

                # ✅ FIX: Check if res is valid before concatenating
                if res is not None and not res.empty:
                    res_final = pd.concat([res_final, res], ignore_index=True)
                else:
                    logger.warning("No transactions found after categorization (empty DataFrame)")
                
            except Exception as file_error:
                error_msg = f"Error processing file: {str(file_error)}"
                logger.error(error_msg)
                # NEW: Update status to 'failed' if file processing fails
                if import_id:
                    update_statement_status(import_id, 'failed', error=error_msg)
                raise  # Re-raise to be caught by outer try-except

        # convert response to webhook event type
        # invoke webhook event
        logger.info("LLM invocation done. Converting df to event list")
        
        # ✅ FIX: Handle empty DataFrame case
        if res_final.empty:
            logger.warning("No transactions found in final result. Statement may have no transactions above threshold.")
            if import_id:
                update_statement_status(import_id, 'completed', error="No transactions found above threshold (150)")
            return res_final
        
        event_list = df_to_event_list(res_final, client_id, accountant_id, file_id)  # ✅ Pass file_id
        
        # ✅ FIX: Only invoke webhook if we have events
        if event_list:
            webhook_response = invoke_webhook(event_list)

            # ✅ FIXED: Update status to 'completed' or 'failed' after processing
            if import_id:
                if webhook_response is not None:
                    if webhook_response.status_code == 200:
                        web_status = 'completed'
                        web_error = None
                    elif webhook_response.status_code == 408:  # ✅ FIX: Timeout - don't mark as failed
                        # Don't mark as failed on timeout - webhook may still process transactions
                        web_status = 'completed'
                        web_error = "Webhook call timed out but transactions may still be processed"
                        logger.warning("Webhook timed out but marking as completed (transactions may still be processed)")
                    else:
                        web_status = 'failed'
                        web_error = webhook_response.text if hasattr(webhook_response, 'text') else "Webhook call failed"
                else:
                    web_status = 'failed'
                    web_error = "Webhook call failed (no response)"
                
                update_statement_status(import_id, web_status, web_error)
                logger.info(f"✅ Updated statement {import_id} to {web_status} status")
        else:
            # No events to send, but processing was successful
            logger.info("No events to send to webhook (empty event list)")
            if import_id:
                update_statement_status(import_id, 'completed', error="No transactions found above threshold")
                logger.info(f"✅ Updated statement {import_id} to 'completed' status (no transactions)")

        return res_final
        
    except Exception as e:
        error_msg = f"Error in classifier_main: {str(e)}"
        logger.error(error_msg)
        # NEW: Update status to 'failed' on any error
        if import_id:
            try:
                update_statement_status(import_id, 'failed', error=error_msg)
            except Exception as status_error:
                logger.error(f"Failed to update status to 'failed': {status_error}")
        raise  # Re-raise the exception


# UPDATED: Webhook endpoint - Removed statement_import_id from transactions, uses junction table only
@app.post("/transactions/webhook")
async def webhook_events(request: Request):
    body = await request.body()
    signature = request.headers.get("x-signature")
    expected = generate_hmac_sha256_signature(os.getenv("WEBHOOK_SIGNATURE_KEY", ""), body.decode())

    if not signature or signature != expected:
        logger.error("Invalid or missing signature for incoming webhook")
        return {"status": "invalid signature"}

    try:
        data = json.loads(body)
        events = data.get("events", [])
    except Exception as e:
        logger.error("Failed to parse incoming webhook JSON: %s", str(e))
        return {"status": "invalid json"}

    logger.info("Received %d events", len(events))

    rows = []
    rows_to_insert = []
    links_to_create = []  # Track links to create in junction table

    # ---- Process events ----
    for event in events:
        amount = event["tx_amount"]
        tx_type = "income" if amount > 0 else "expense" if amount < 0 else "transfer"
        raw_date = event["tx_timestamp"]

        # Convert date safely
        try:
            dt = datetime.strptime(raw_date, "%d/%m/%Y")
        except ValueError:
            try:
                dt = datetime.fromisoformat(raw_date)
            except ValueError:
                logger.error("Bad date format in event: %s", raw_date)
                continue  # only continue HERE inside the loop

        occurred_at = dt.strftime("%Y-%m-%d")

        # UPDATED: Removed statement_import_id from transaction_data
        # Relationships are now only tracked via statement_transactions junction table
        transaction_data = {
            "user_id": event["client_id"],
            "source": "statement",
            "amount": amount,
            "currency": "INR",
            "type": tx_type,
            "raw_description": event["tx_narration"],
            "merchant": None,
            "status": "final",
            "category_ai_id": event["category_id"],
            "category_user_id": None,
            # statement_import_id removed - relationships now only in junction table
            "occurred_at": occurred_at,
        }
        
        rows.append({
            "transaction_data": transaction_data,
            "file_id": event.get("file_id")  # Keep file_id separate for junction table
        })

    # After loop completes
    if not rows:
        logger.info("No rows to insert")
        return {"status": "no events"}

    # UPDATED: GLOBAL deduplication - Check across ALL statement files
    # This prevents duplicate transactions when users upload overlapping statements
    logger.info(f"Checking for duplicates among {len(rows)} transactions (GLOBAL check across all files)")
    
    for row_item in rows:
        transaction_data = row_item["transaction_data"]
        current_file_id = row_item["file_id"]
        
        # UPDATED: Query no longer includes statement_import_id (column doesn't exist)
        # Match by: user_id, raw_description, amount, occurred_at, source
        # This ensures the same transaction from different statement files is only stored once
        query = supabase.table("transactions").select("id, category_ai_id, category_user_id").eq("user_id", transaction_data["user_id"]).eq("raw_description", transaction_data["raw_description"]).eq("amount", transaction_data["amount"]).eq("occurred_at", transaction_data["occurred_at"]).eq("source", "statement")
        
        existing = query.execute()
        
        if existing.data and len(existing.data) > 0:
            # Transaction already exists (from any statement file) - skip insertion
            existing_id = existing.data[0]["id"]
            
            logger.info(f"Duplicate transaction found (ID: {existing_id}). Creating link in junction table if needed.")
            logger.info(f"   Details: {transaction_data['raw_description'][:50]}... | Amount: {transaction_data['amount']} | Date: {transaction_data['occurred_at']}")
            
            # Create link in junction table (even if duplicate from different file)
            # This ensures the statement can find all its transactions
            if current_file_id:
                links_to_create.append({
                    "statement_import_id": current_file_id,
                    "transaction_id": existing_id
                })
                logger.info(f"Added link: statement {current_file_id} -> transaction {existing_id}")
            
            # OPTIONAL: Update existing transaction if category_ai_id changed
            # Only update if category_ai_id is different and category_user_id is still null
            # (Don't overwrite user's manual category assignment)
            try:
                existing_cat_ai = existing.data[0].get("category_ai_id")
                existing_cat_user = existing.data[0].get("category_user_id")
                
                # Only update if:
                # 1. category_ai_id is different AND
                # 2. category_user_id is null (user hasn't manually set a category)
                if existing_cat_ai != transaction_data["category_ai_id"] and existing_cat_user is None:
                    supabase.table("transactions").update({
                        "category_ai_id": transaction_data["category_ai_id"]
                    }).eq("id", existing_id).execute()
                    logger.info(f"Updated category_ai_id for existing transaction {existing_id}")
            except Exception as update_error:
                logger.warning(f"Could not update existing transaction: {update_error}")
        else:
            # Transaction doesn't exist globally - add to insert list
            rows_to_insert.append({
                "transaction_data": transaction_data,
                "file_id": current_file_id
            })

    # Insert new transactions and create links
    inserted_transaction_ids = []
    
    if rows_to_insert:
        logger.info(f"Inserting {len(rows_to_insert)} new transactions (skipped {len(rows) - len(rows_to_insert)} duplicates)")
        
        # Extract just transaction_data for insert
        transactions_to_insert = [item["transaction_data"] for item in rows_to_insert]
        
        try:
            resp = supabase.table("transactions").insert(transactions_to_insert).execute()
            logger.info(f"Successfully inserted {len(rows_to_insert)} transactions")
            
            if resp.get("error"):
                logger.error("Failed to insert transactions: %s", resp["error"])
                return {"status": "error", "message": str(resp["error"])}
            
            # Extract IDs of newly inserted transactions
            if resp.data:
                inserted_transaction_ids = [tx["id"] for tx in resp.data]
                
                # Create links for newly inserted transactions
                for i, row_item in enumerate(rows_to_insert):
                    file_id = row_item["file_id"]
                    if file_id and i < len(inserted_transaction_ids):
                        links_to_create.append({
                            "statement_import_id": file_id,
                            "transaction_id": inserted_transaction_ids[i]
                        })
                        logger.info(f"Added link: statement {file_id} -> transaction {inserted_transaction_ids[i]}")
            
        except Exception as insert_error:
            logger.error(f"Exception during insert: {insert_error}")
            return {"status": "error", "message": str(insert_error)}
    else:
        logger.info("All transactions already exist. No new transactions to insert.")

    # Create all links in junction table
    if links_to_create:
        logger.info(f"Creating {len(links_to_create)} links in statement_transactions junction table")
        try:
            # Use upsert to handle duplicate links gracefully (UNIQUE constraint)
            for link in links_to_create:
                try:
                    supabase.table("statement_transactions").upsert(
                        link,
                        on_conflict="statement_import_id,transaction_id"
                    ).execute()
                    logger.info(f"Created link: statement {link['statement_import_id']} -> transaction {link['transaction_id']}")
                except Exception as link_error:
                    # Link might already exist (from previous processing), that's OK
                    logger.debug(f"Link already exists or error creating link: {link_error}")
            
            logger.info(f"Successfully created {len(links_to_create)} links in junction table")
        except Exception as link_error:
            logger.warning(f"Some links could not be created: {link_error}")
            # Don't fail the entire operation if link creation fails
            # The transactions are still inserted/updated

    return {
        "status": "success",
        "inserted": len(rows_to_insert),
        "skipped": len(rows) - len(rows_to_insert),
        "links_created": len(links_to_create)
    }