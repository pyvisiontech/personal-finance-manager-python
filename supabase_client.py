from supabase import create_client
import os

# Your Supabase URL and API key
SUPABASE_URL = "https://eoyrvbmyvvomwkzxatrg.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_PRIVATE_KEY")

def fetch_supabase_db(client_id):
  # Initialize client
  supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

  primary_key_value = client_id

  response = supabase.table('clients').select('*').eq('id', primary_key_value).execute()
  if response.data:
    client_info = response.data[0]
    return client_info
  else:
    return -1;


def fetch_supabase_cat_db():
  # Initialize client
  supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

  response = supabase.table('categories').select('*').execute()
  if response.data:
    client_info = response.data
    return client_info
  else:
    return -1;

def upsert_category(category_name):
    """
    Upsert (insert or update) categories by name only.
    category_name: string (category name)
    """
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Prepare list of dicts for upsert, matching table columns
    rows = [{"name": category_name}]
    response = supabase.table('categories').upsert(rows).execute()
    if response.data:
        return response.data
    else:
        return None
