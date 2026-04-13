from fastapi import FastAPI, Request, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import pandas as pd
from openai import OpenAI
import tiktoken
import json
import re
import fitz
import pymupdf4llm
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
from cryptography.fernet import Fernet

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

# Field-level encryption for raw_description (at rest)
FERNET_KEY = os.getenv("FERNET_SECRET_KEY")
if FERNET_KEY:
    try:
        cipher_suite = Fernet(FERNET_KEY.encode())
    except Exception as e:
        logger.warning("FERNET_SECRET_KEY invalid: %s. Falling back to plain text.", e)
        cipher_suite = None
else:
    logger.warning("FERNET_SECRET_KEY not set. Falling back to plain text.")
    cipher_suite = None


def encrypt_string(text: str) -> str:
    if not text or not cipher_suite:
        return text or ""
    return cipher_suite.encrypt((text or "").encode("utf-8")).decode("utf-8")


def decrypt_string(encrypted_text: str) -> str:
    if not encrypted_text or not cipher_suite:
        return encrypted_text or ""
    try:
        return cipher_suite.decrypt(encrypted_text.encode("utf-8")).decode("utf-8")
    except Exception:
        return encrypted_text  # Fallback if not encrypted (e.g. legacy rows)


security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    secret = os.getenv("SUPABASE_JWT_SECRET")
    if not secret:
        logger.error("SUPABASE_JWT_SECRET is not set in environment")
        raise HTTPException(status_code=500, detail="Server misconfiguration")
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="authenticated",
            options={"verify_aud": True},
        )
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except jwt.InvalidAudienceError:
        logger.warning("JWT audience invalid, retrying without audience check")
        try:
            payload = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
            return payload["sub"]
        except Exception as e2:
            logger.warning("JWT decode failed without audience: %s", e2)
            raise HTTPException(status_code=401, detail="Invalid authentication token")
    except Exception as e:
        logger.warning("JWT verification failed: %s - %s", type(e).__name__, str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


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


# PII redaction before sending text to LLM
def upload_debug_to_supabase(filename: str, content: str):
    """
    Uploads debug text content to Supabase Storage bucket 'debug-dumps'.
    Enables remote auditing of AI outputs without direct server access.
    """
    try:
        file_data = content.encode('utf-8')
        # We use upsert=True to overwrite the previous 'last' file for cleaner auditing
        supabase.storage.from_('debug-dumps').upload(
            path=filename,
            file=file_data,
            file_options={"cache-control": "3600", "upsert": "true"}
        )
        logger.info(f"📁 ✅ Debug dump uploaded to Supabase Storage: {filename}")
    except Exception as e:
        logger.error(f"📁 ❌ Failed to upload debug dump '{filename}' to Supabase: {e}")

def redact_pii(text: str) -> str:
    # 1. Redact Indian PAN Card Numbers (e.g. ABCDE1234F)
    text = re.sub(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', '[PAN_REDACTED]', text)
    
    # 2. Redact Account Numbers (typically 9 to 18 contiguous digits)
    # We use lookarounds to avoid redacting legitimate small amounts or dates
    text = re.sub(r'\b\d{9,18}\b', '[ACCOUNT_REDACTED]', text)
    
    # 3. Redact common Indian Phone Numbers (10 digits starting with 6-9, optionally with +91)
    text = re.sub(r'(?:(?:\+|0{0,2})91(\s*[\-]\s*)?|[0]?)?[6789]\d{9}', '[PHONE_REDACTED]', text)

    # 4. Redact Aadhar Numbers (12 digits separated by spaces or continuous)
    text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[AADHAR_REDACTED]', text)
    
    return text


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

def categorize_transactions_batch(client, df, amount_threshold=0, batch_size=20, model="gpt-4o-mini", person_name='Abhishek', mobile_numbers='7206527787'):
    """
    Categorize transactions using LLM in batches with confidence scores.
    Handles 'Narration', 'Credit Amount', 'Debit Amount', and 'Date' columns.
    Returns df with new 'Amount', 'Category', 'Confidence' columns.
    """
    results = []

    # ✅ FIX: Convert columns to numeric, removing any commas or weird characters first
    df["Credit Amount"] = pd.to_numeric(df["Credit Amount"].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df["Debit Amount"] = pd.to_numeric(df["Debit Amount"].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # Unified Amount column
    # ✅ FIX: Ensure numeric types before subtraction to avoid crash
    df["Credit Amount"] = pd.to_numeric(df["Credit Amount"].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df["Debit Amount"] = pd.to_numeric(df["Debit Amount"].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    df["Amount"] = df["Credit Amount"] - df["Debit Amount"]
    
    # ✅ DEDUPE FIX: Normalize description (lowercase/strip) for consistent hashing
    df.rename(columns={"Narration": "Description"}, inplace=True)
    df["Description"] = df["Description"].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

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
        - CRITICAL RULE: DO NOT modify the sign (+ or -) or value of the Amount. Return it EXACTLY as provided. Do not alter the Date or Description.
        
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
        logger.info(f"Categorization Stage response: {response}")
        raw_output = response.choices[0].message.content
        
        # ✅ DEBUG AUDIT: Save and upload the raw JSON
        try:
            with open("debug_categorization_json.json", "w", encoding="utf-8") as f:
                f.write(raw_output)
            upload_debug_to_supabase("last_categorization.json", raw_output)
            logger.info("📁 Debug: Categorization JSON saved and uploaded")
        except Exception as e:
            logger.error(f"Failed to handle debug JSON: {e}")

        logger.info("✅ STAGE 2: LLM categorization JSON generated")
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
    """
    Extract text from PDF using pymupdf4llm which converts to Markdown.
    This preserves table structure (Date, Narration, Debit, Credit columns)
    much better than plain text extraction, improving LLM parsing accuracy.
    """
    markdown_text = pymupdf4llm.to_markdown(doc)
    return markdown_text


def download_file_from_s3(presigned_url):
    response = requests.get(presigned_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF: {response.status_code}")

    return response


def pdf_to_csv(file_response, client, model):
    pdf_bytes = file_response.content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = extract_text(doc)
    
    # NEW STEP: Scrub PII
    safe_text = redact_pii(extracted_text)

    # Construct the prompt
    prompt = f"""
        You are an AI assistant specialized in parsing bank statements with complex layouts.
        The input text below is a "Flow" extraction where rows might be broken across several lines. 
        You MUST reconstruct each transaction by following these steps:
        1. ANCHOR: Locate the Serial Number (S No) (e.g., 1, 2, 3...).
        2. DATE: The Transaction Date is the date appearing immediately after or next to the S No.
        3. NARRATION: The text following the date is the Transaction Remarks/Narration.
        4. AMOUNT MAPPING: At the bottom of each section, there is a list of numbers. These are [Amount] [Balance] pairs. 
           The 1st S No corresponds to the 1st pair in that list. The 10th S No corresponds to the 10th pair.
        5. IGNORE BALANCE: The second number in each pair is the 'Balance'. You MUST IGNORE it.
        6. OUTPUT: Date,Narration,Debit Amount,Credit Amount.

        Guidelines:
        - Output ONLY raw CSV, NO explanations or Markdown.
        - The FIRST LINE must be the header: Date,Narration,Debit Amount,Credit Amount
        - DATE FORMAT RULE: Output as YYYY-MM-DD. Use '2026' if the year is missing.
        - RELIABLE EXTRACTION: Look closely at the list of numbers at the end of the text to find the correct amount for each S No.
        - CRITICAL RULE: Map 'Withdrawal' to 'Debit Amount', and 'Deposit' to 'Credit Amount'.

        Here is the extracted text:
        {safe_text}
    """
    
    # ✅ DEBUG AUDIT: Save and upload the raw Markdown
    try:
        with open("debug_raw_pdf_text.md", "w", encoding="utf-8") as f:
            f.write(safe_text)
        upload_debug_to_supabase("last_pdf_markdown.md", safe_text)
        logger.info(f"📁 Debug: Raw PDF Markdown saved and uploaded. Preview:\n{safe_text[:3000]}")
    except Exception as e:
        logger.error(f"Failed to handle debug markdown: {e}")

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
    logger.info("✅ LLM parsed PDF to CSV successfully")
    
    # ✅ DEBUG AUDIT: Save and upload the raw CSV
    try:
        with open("debug_pdf_to_csv_output.csv", "w", encoding="utf-8") as f:
            f.write(csv_output)
        upload_debug_to_supabase("last_pdf_extraction.csv", csv_output)
        logger.info("📁 Debug: LLM CSV saved and uploaded")
    except Exception as e:
        logger.error(f"Failed to handle debug CSV: {e}")

    # Save to CSV
    with open("bank_statement_parsed.csv", "w", encoding="utf-8") as f:
        f.write(csv_output)
    
    try:
        df = pd.read_csv("bank_statement_parsed.csv")
    except Exception as e:
        logger.error(f"❌ Failed to read LLM CSV output: {e}")
        return pd.DataFrame()
    
    # ✅ STABILITY FIX: Ensure Debit/Credit are numeric (removes commas/strings)
    df["Credit Amount"] = pd.to_numeric(df["Credit Amount"].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df["Debit Amount"] = pd.to_numeric(df["Debit Amount"].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # ✅ DATE FIX: Ensure YYYY-MM-DD format regardless of AI output
    try:
        # We try to parse whatever the AI gave us and force it to YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    except Exception as e:
        logger.warning(f"⚠️ Date normalization failed for some rows: {e}")

    logger.info(f"📊 PDF Step Complete: Found {len(df)} transactions")
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
async def classifier_api(request: ClassifierRequest, user_id: str = Depends(verify_token)):
    """
    UPDATED: Now handles status updates throughout the processing lifecycle.
    user_id comes from verified JWT; we do not trust client_id from request body.
    Idempotent: if statement is already 'completed' or 'processing', skip to prevent double processing.
    """
    import_id = request.import_id
    client_id = user_id  # Use cryptographically verified user_id from token
    
    # Idempotency: skip if this statement is already completed or currently processing
    if import_id:
        try:
            r = supabase.table("statement_imports").select("status").eq("id", import_id).execute()
            if r.data and len(r.data) > 0:
                current_status = (r.data[0].get("status") or "").strip().lower()
                if current_status == "completed":
                    logger.info(f"Statement {import_id} already completed; skipping duplicate request")
                    return {"status": "completed"}
                if current_status == "processing":
                    logger.info(f"Statement {import_id} already processing; skipping duplicate request")
                    return {"status": "completed"}
        except Exception as e:
            logger.warning(f"Could not check statement status for idempotency: {e}")
            # Proceed with processing if we can't read status
    
    # If import_id is provided, update status to 'processing' immediately
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
        logger.info(f"🚀 ENGINE START: Using model '{model}' with DeepSeek base URL")

        for file in file_list:
            try:
                if "pdf" in file.headers.get("Content-Type", ""):
                    df = pdf_to_csv(file, client, model)
                    logger.info("✅ STAGE 1: PDF Extraction complete")
                    
                    # ✅ QUALITY FIX: Use batch_size=20 (matching successful local tests)
                    res = categorize_transactions_batch(
                        client, df, amount_threshold=0, batch_size=20, 
                        model=model, person_name=name, mobile_numbers=mob_no
                    )
                    logger.info(f"✅ STAGE 2: Categorization complete ({len(res)} transactions)")
                else:
                    df = pd.read_csv(io.StringIO(file.content.decode('utf-8')))
                    ## columns intent
                    map = csv_col_identify(df.columns, client, model)
                    logger.info(f"columns from the csv files: {map}")
                    df = df.rename(columns=map)
                    # Step 2: Keep only columns present in mapping
                    df = df[[col for col in map.values() if col in df.columns]]
                    
                    # ✅ QUALITY FIX: Use batch_size=20
                    res = categorize_transactions_batch(
                        client, df, amount_threshold=0, batch_size=20, 
                        model=model, person_name=name, mobile_numbers=mob_no
                    )
                    logger.info(f"✅ CATEGORY STEP: Categorized {len(res)} transactions via {model}")

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
        
        # Deduplicate by transaction identity so duplicate rows from PDF/LLM don't create duplicate events
        cols = [c for c in ['Description', 'Amount', 'Date'] if c in res_final.columns]
        if cols:
            before = len(res_final)
            res_final = res_final.drop_duplicates(subset=cols, keep='first')
            if len(res_final) < before:
                logger.info(f"Dropped {before - len(res_final)} duplicate rows from DataFrame before webhook")
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


# GET /transactions — returns decrypted raw_description (Option 1: server-side decryption for clients)
@app.get("/transactions")
async def get_transactions(
    user_id: str = Depends(verify_token),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category_id: Optional[str] = None,
    group_id: Optional[str] = None,
    statement_id: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Fetch transactions for the authenticated user (or group), with raw_description decrypted.
    Query params: start_date, end_date, category_id, group_id, statement_id, limit.
    """
    select_cols = "id, user_id, source, amount, currency, type, raw_description, raw_description_hash, merchant, status, category_ai_id, category_user_id, occurred_at, created_at, category_user:category_user_id(id, name, icon), category_ai:category_ai_id(id, name, icon)"
    user_ids = [user_id]

    if statement_id:
        # Verify statement belongs to user and get linked transaction IDs
        st = supabase.table("statement_imports").select("id, user_id").eq("id", statement_id).execute()
        if not st.data or st.data[0].get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Statement not found")
        junc = supabase.table("statement_transactions").select("transaction_id").eq("statement_import_id", statement_id).execute()
        tx_ids = [r["transaction_id"] for r in (junc.data or [])]
        if not tx_ids:
            return []
        q = supabase.table("transactions").select(select_cols).in_("id", tx_ids)
        if start_date:
            q = q.gte("occurred_at", start_date)
        if end_date:
            q = q.lte("occurred_at", end_date)
        if category_id:
            q = q.eq("category_user_id", category_id)
        if limit is not None and limit > 0:
            q = q.limit(limit)
        resp = q.order("occurred_at", desc=True).execute()
    else:
        q = supabase.table("transactions").select(select_cols)
        if group_id:
            members = supabase.table("group_members").select("user_id").eq("group_id", group_id).execute()
            if not members.data:
                return []
            user_ids = [m["user_id"] for m in members.data]
            q = q.in_("user_id", user_ids)
        else:
            q = q.eq("user_id", user_id)
        if start_date:
            q = q.gte("occurred_at", start_date)
        if end_date:
            q = q.lte("occurred_at", end_date)
        if category_id:
            q = q.eq("category_user_id", category_id)
        if limit is not None and limit > 0:
            q = q.limit(limit)
        resp = q.order("occurred_at", desc=True).execute()

    rows = resp.data or []
    if limit is not None and limit > 0 and len(rows) > limit:
        rows = rows[:limit]

    # Decrypt raw_description for each row (legacy plaintext rows pass through decrypt_string)
    for r in rows:
        if r.get("raw_description"):
            r["raw_description"] = decrypt_string(r["raw_description"])
    return rows


# UPDATED: Webhook endpoint - Removed statement_import_id from transactions, uses junction table only
# ✅ FIXED: Do not use resp.get("error") - APIResponse has .data only; use resp.data for inserted IDs
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
        dt = None
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"):
            try:
                dt = datetime.strptime(raw_date, fmt)
                break
            except (TypeError, ValueError):
                continue
        if dt is None:
            try:
                # Handles ISO formats like 2026-02-03T00:00:00
                dt = datetime.fromisoformat(raw_date)
            except (TypeError, ValueError):
                logger.error("Bad date format in event: %s", raw_date)
                continue  # only continue HERE inside the loop

        occurred_at = dt.strftime("%Y-%m-%d")

        # Encrypt sensitive narrative and compute hash for fast deduplication (no decrypt at insert)
        tx_narration = (event.get("tx_narration") or "").strip()
        narration_hash = hashlib.sha256(tx_narration.encode("utf-8")).hexdigest()

        # UPDATED: Removed statement_import_id from transaction_data
        # Relationships are now only tracked via statement_transactions junction table
        transaction_data = {
            "user_id": event["client_id"],
            "source": "statement",
            "amount": amount,
            "currency": "INR",
            "type": tx_type,
            "raw_description": encrypt_string(tx_narration),
            "raw_description_hash": narration_hash,
            "merchant": None,
            "status": "final",
            "category_ai_id": event["category_id"],
            "category_user_id": None,
            # statement_import_id removed - relationships now only in junction table
            "occurred_at": occurred_at,
        }
        
        rows.append({
            "transaction_data": transaction_data,
            "file_id": event.get("file_id"),  # Keep file_id separate for junction table
            "plain_narration": tx_narration,   # For legacy dedupe check only (not stored)
        })

    # After loop completes
    if not rows:
        logger.info("No rows to insert")
        return {"status": "no events"}

    # UPDATED: GLOBAL deduplication - Check across ALL statement files
    # This prevents duplicate transactions when users upload overlapping statements
    logger.info(f"Checking for duplicates among {len(rows)} transactions (GLOBAL check across all files)")
    
    # Within-request dedupe: same payload can have duplicate events; only insert one per (hash, amount, date)
    seen_in_batch = set()
    
    for row_item in rows:
        transaction_data = row_item["transaction_data"]
        current_file_id = row_item["file_id"]
        plain_narration = row_item.get("plain_narration") or ""

        batch_key = (transaction_data["raw_description_hash"], transaction_data["amount"], transaction_data["occurred_at"])
        if batch_key in seen_in_batch:
            logger.debug(f"Skipping duplicate event in same request: amount={transaction_data['amount']} date={transaction_data['occurred_at']}")
            continue
        seen_in_batch.add(batch_key)

        # Dedupe by hash first (fast; no decryption). Fallback to raw_description for legacy rows.
        query_by_hash = (
            supabase.table("transactions")
            .select("id, category_ai_id, category_user_id")
            .eq("user_id", transaction_data["user_id"])
            .eq("raw_description_hash", transaction_data["raw_description_hash"])
            .eq("amount", transaction_data["amount"])
            .eq("occurred_at", transaction_data["occurred_at"])
            .eq("source", "statement")
        )
        existing = query_by_hash.execute()

        if (not existing.data or len(existing.data) == 0) and plain_narration:
            # Legacy: rows inserted before hash existed have no raw_description_hash; match by plaintext
            query_legacy = (
                supabase.table("transactions")
                .select("id, category_ai_id, category_user_id")
                .eq("user_id", transaction_data["user_id"])
                .eq("raw_description", plain_narration)
                .eq("amount", transaction_data["amount"])
                .eq("occurred_at", transaction_data["occurred_at"])
                .eq("source", "statement")
            )
            existing = query_legacy.execute()

        if existing.data and len(existing.data) > 0:
            # Transaction already exists (from any statement file) - skip insertion
            existing_id = existing.data[0]["id"]
            logger.info(f"Duplicate transaction found (ID: {existing_id}). Creating link in junction table if needed.")
            logger.info(f"   Hash: {transaction_data['raw_description_hash'][:16]}... | Amount: {transaction_data['amount']} | Date: {transaction_data['occurred_at']}")
            
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

            # ✅ FIXED: Do NOT use resp.get("error") - Supabase returns APIResponse with .data, not a dict.
            # Errors are raised as exceptions; we only need resp.data for inserted row IDs.
            if resp.data:
                inserted_transaction_ids = [tx["id"] for tx in resp.data]
            else:
                inserted_transaction_ids = []

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