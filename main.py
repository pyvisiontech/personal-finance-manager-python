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


# ============================================================================
# ✅ NEW FUNCTION: Update statement status in Supabase
# ============================================================================
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
    """
    raw = raw.strip()
    # Remove ```json or ``` markdown if present
    raw = re.sub(r"^```json", "", raw)
    raw = re.sub(r"```$", "", raw)

    # Try normal parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if raw.startswith("[") and not raw.endswith("]"):
            raw += "]"
        try:
            return json.loads(raw)
        except:
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


# -------------------- Core processing functions --------------------

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
            logger.info("Success to parse JSON for the batch from main classifier")
        else:
            print("⚠️ Failed to parse JSON for batch. Raw output:", raw_output)
            logger.info("Failed to parse JSON for the batch from main classifier")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

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

    # Remove markdown if present (like ```json ... ```)
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    # Parse into dictionary safely
    try:
        mapping = json.loads(raw_output)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON from LLM output. Raw output:")
        print(raw_output)
        mapping = {}

    return mapping


def df_to_event_list(df, client_id, accountant_id):
    """
    Convert DataFrame to event list for webhook.
    Removed file_id parameter as it's no longer needed.
    """
    # Select required columns and convert to list of dicts
    required_cols = ['Category', 'Confidence', 'Reason', 'Description', 'Amount', 'Date']
    event_list = df[required_cols].to_dict(orient='records')

    cat_id_map = fetch_supabase_cat_db()
    name_to_id_map = {item['name']: item['id'] for item in cat_id_map}
    # Add additional info to each event object
    for event in event_list:
        event['client_id'] = client_id
        event['accountant_id'] = accountant_id
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


# -------------------- Webhook HMAC helpers & invoker --------------------

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
    webhook_url = "https://statement-classifier-python-2.onrender.com/transactions/webhook"


    signature = generate_hmac_sha256_signature(os.getenv("WEBHOOK_SIGNATURE_KEY", ""), raw_json_string)
    headers = {"Content-Type": "application/json", "x-signature": signature}

    logger.info("Invoking webhook event to %s", webhook_url)
    try:
        response = requests.post(webhook_url, data=raw_json_string, headers=headers, timeout=30)
    except Exception as e:
        logger.error("Exception while invoking webhook: %s", str(e))
        return

    if response.status_code == 200:
        logger.info("Webhook success")
        print("Webhook successfully invoked.")
    else:
        logger.error(f"Webhook invocation failed with status code {response.status_code} - Response: {response.text}")
        print(f"Webhook invocation failed: {response.status_code} - {response.text}")


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


# ============================================================================
# ✅ UPDATED: ClassifierRequest - Removed file_id field
# ============================================================================
class ClassifierRequest(BaseModel):
    import_id: Optional[str] = None  # ✅ import_id from statement_imports table
    client_id: str
    signed_url: str
    accountant_id: Optional[str] = None
    # Keep backward compatibility with old fields
    user_id: Optional[str] = None  # ✅ Also accept user_id (same as client_id)
    file_url: Optional[str] = None  # ✅ Also accept file_url (alternative to signed_url)


@app.post("/classifier")
async def classifier_api(request: ClassifierRequest):
    """
    ✅ UPDATED: Now handles status updates throughout the processing lifecycle
    Removed file_id from processing
    """
    import_id = request.import_id
    client_id = request.client_id or request.user_id  # Support both field names
    
    # ✅ NEW: If import_id is provided, update status to 'processing' immediately
    if import_id:
        try:
            update_statement_status(import_id, 'processing', processor_job_id=f"job_{import_id[:8]}")
            logger.info(f"✅ Updated statement {import_id} to 'processing' status")
        except Exception as e:
            logger.error(f"⚠️ Failed to update status to 'processing': {e}")
            # Continue processing even if status update fails
    
    file_list = []
    
    # ✅ NEW: Support both signed_url and file_url
    file_url = request.signed_url or request.file_url
    
    try:
        # download files using presigned urls
        file_list.append(download_file_from_s3(file_url))
        
        # fetch client info from supabase db
        client_info = fetch_supabase_db(client_id)
        
        if client_info == -1:
            error_msg = f"Client not found: {client_id}"
            logger.error(error_msg)
            # ✅ NEW: Update status to 'failed' if client not found
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
            import_id  # ✅ Pass import_id to classifier_main
        )
        
        # ✅ NEW: Status will be updated to 'completed' or 'failed' in classifier_main
        return {"status": "completed"}
        
    except Exception as e:
        error_msg = f"Error processing statement: {str(e)}"
        logger.error(error_msg)
        # ✅ NEW: Update status to 'failed' on any exception
        if import_id:
            try:
                update_statement_status(import_id, 'failed', error=error_msg)
            except Exception as status_error:
                logger.error(f"Failed to update status to 'failed': {status_error}")
        return {"status": "error", "message": error_msg}


# ============================================================================
# ✅ UPDATED: classifier_main - Removed file_id parameter
# ============================================================================
async def classifier_main(file_list, name, mob_no, client_id, accountant_id, import_id=None):
    """
    ✅ UPDATED: Now accepts import_id and updates status to 'completed' or 'failed'
    Removed file_id parameter as it's no longer needed
    """
    res_final = pd.DataFrame()
    
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

                res_final = pd.concat([res_final, res], ignore_index=True)
                
            except Exception as file_error:
                error_msg = f"Error processing file: {str(file_error)}"
                logger.error(error_msg)
                # ✅ NEW: Update status to 'failed' if file processing fails
                if import_id:
                    update_statement_status(import_id, 'failed', error=error_msg)
                raise  # Re-raise to be caught by outer try-except

        # convert response to webhook event type
        # invoke webhook event
        logger.info("LLM invocation done. Converting df to event list")
        event_list = df_to_event_list(res_final, client_id, accountant_id)  # ✅ Removed file_id
        invoke_webhook(event_list)

        # ✅ NEW: Update status to 'completed' after successful processing
        if import_id:
            update_statement_status(import_id, 'completed')
            logger.info(f"✅ Updated statement {import_id} to 'completed' status")

        return res_final
        
    except Exception as e:
        error_msg = f"Error in classifier_main: {str(e)}"
        logger.error(error_msg)
        # ✅ NEW: Update status to 'failed' on any error
        if import_id:
            try:
                update_statement_status(import_id, 'failed', error=error_msg)
            except Exception as status_error:
                logger.error(f"Failed to update status to 'failed': {status_error}")
        raise  # Re-raise the exception


# -------------------- NEW WEBHOOK RECEIVER ENDPOINT --------------------

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

    # ---- FIXED LOOP ----
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

        rows.append({
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
            "occurred_at": occurred_at,
        })

    # After loop completes
    if not rows:
        logger.info("No rows to insert")
        return {"status": "no events"}

    resp = supabase.table("transactions").insert(rows).execute()
    logger.info("Insert response: %s", resp)

    if resp.get("error"):
        logger.error("Failed to insert transactions: %s", resp["error"])
        return {"status": "error"}

    return {"status": "success"}

