import os
import base64
import pdb
import re

from google.auth.transport.requests import Request # type: ignore
from google.oauth2.credentials import Credentials # type: ignore
from google_auth_oauthlib.flow import InstalledAppFlow # type: ignore
from googleapiclient.discovery import build # type: ignore
from googleapiclient.errors import HttpError # type: ignore

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
GMAIL_CREDS_LOCAL_PATH = '/Users/lakshayk/Developer/Tripsy/Tripsy/tools/email_tool/'

def authenticate_gmail_api():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(os.path.join(GMAIL_CREDS_LOCAL_PATH, 'token.json')):
        creds = Credentials.from_authorized_user_file(os.path.join(GMAIL_CREDS_LOCAL_PATH, 'token.json'), SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Ensure credentials.json is in the same directory
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.join(GMAIL_CREDS_LOCAL_PATH, 'credentials.json'), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds

def build_query(labels=['VACATION'], start_date=None, end_date=None, title_keywords=None, content_keywords=None):
    """
    Builds the Gmail API query string based on provided filters.

    Args:
        labels (list, optional): List of labels to filter by (e.g., ['INBOX', 'UNREAD']). Defaults to None.
        start_date (str, optional): Start date for filtering (YYYY/MM/DD). Defaults to None.
        end_date (str, optional): End date for filtering (YYYY/MM/DD). Defaults to None.
        title_keywords (list, optional): List of keywords to search in the email title. Defaults to None.
        content_keywords (list, optional): List of keywords to search in the email content. Defaults to None.

    Returns:
        str: The constructed Gmail API query string.
    """
    query_parts = []

    if labels:
        query_parts.append('label:VACATION') # Default label is 'VACATION'

    if start_date:
        query_parts.append(f'after:{start_date}')

    if end_date:
        query_parts.append(f'before:{end_date}')

    # Modified to use OR logic for title_keywords
    if title_keywords:
        # Title keywords use 'subject:' operator and are combined with OR
        subject_query_parts = [f'subject:"{keyword}"' for keyword in title_keywords]
        query_parts.append(' OR '.join(subject_query_parts))

    # if content_keywords:
    #     # Content keywords are searched in the full text and combined with AND
    #     query_parts.extend(content_keywords) # Keywords are treated as AND by default

    # Combine all parts with spaces (which acts as AND in Gmail API)
    return ' '.join(query_parts)

def get_email_parts(message_payload):
    """
    Recursively extracts all parts from a Gmail message payload.

    Args:
        message_payload (dict): The 'payload' part of a Gmail message.

    Returns:
        list: A list of message parts (dictionaries).
    """
    parts = []
    if 'parts' in message_payload:
        for part in message_payload['parts']:
            parts.extend(get_email_parts(part)) # Recursively get parts
    else:
        parts.append(message_payload) # Add the part if it has no nested parts
    return parts


def get_email_body_html_and_plain(message):
    """
    Extracts the HTML and plain text bodies from a Gmail message.
    Prioritizes HTML if available.

    Args:
        message (dict): The Gmail message dictionary.

    Returns:
        tuple: A tuple containing (html_body, plain_text_body).
               Returns (None, None) if no text or html parts are found.
    """
    html_body = None
    plain_text_body = None

    # Get all parts from the message payload
    all_parts = get_email_parts(message['payload'])

    for part in all_parts:
        if part['mimeType'] == 'text/html':
            if 'body' in part and 'data' in part['body']:
                data = part['body']['data']
                html_body = base64.urlsafe_b64decode(data).decode('utf-8')
        elif part['mimeType'] == 'text/plain':
            if 'body' in part and 'data' in part['body']:
                data = part['body']['data']
                plain_text_body = base64.urlsafe_b64decode(data).decode('utf-8')

    # Prioritize HTML if both are available
    return html_body, plain_text_body

def fetch_emails(labels=None, start_date=None, end_date=None, title_keywords=None, content_keywords=None, max_results=1000):
    """
    Fetches emails from Gmail based on specified filters.

    Args:
        labels (list, optional): List of labels to filter by (e.g., ['INBOX', 'UNREAD']). Defaults to None.
        start_date (str, optional): Start date for filtering (YYYY/MM/DD). Defaults to None.
        end_date (str, optional): End date for filtering (YYYY/MM/DD). Defaults to None.
        title_keywords (list, optional): List of keywords to search in the email title. Defaults to None.
        content_keywords (list, optional): List of keywords to search in the email content. Defaults to None.
        max_results (int, optional): Maximum number of emails to fetch. Defaults to 50.

    Returns:
        list: A list of dictionaries, where each dictionary represents an email with
              'id', 'subject', 'from', 'to', 'date', and 'body' fields.
              Returns an empty list if no emails are found or an error occurs.
    """
    creds = authenticate_gmail_api()
    if not creds:
        print("Authentication failed.")
        raise Exception("Authentication failed.")

    try:
        # Build the Gmail service
        service = build('gmail', 'v1', credentials=creds)

        # Build the query string
        query = build_query(labels, start_date, end_date, title_keywords, content_keywords)
        print(f"Using query: {query}") # Print the query for debugging

        # Call the Gmail API to list messages
        results = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        messages = results.get('messages', [])

        fetched_emails = []

        if not messages:
            print('No messages found matching the criteria.')
        else:
            print(f'Found {len(messages)} messages.')
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()

                # Extract relevant information and bodies
                html_body, plain_text_body = get_email_body_html_and_plain(msg)

                # Extract relevant information
                email_data = {
                    'id': msg['id'],
                    'subject': None,
                    'from': None,
                    'to': None,
                    'date': None,
                    # 'html_body' : html_body, # Get the HTML body
                    # 'plain_text_body' : plain_text_body, # Get the plain text body
                }

                # Extract headers
                for header in msg['payload']['headers']:
                    if header['name'] == 'Subject':
                        email_data['subject'] = header['value']
                    elif header['name'] == 'From':
                        email_data['from'] = header['value']
                    elif header['name'] == 'To':
                        email_data['to'] = header['value']
                    elif header['name'] == 'Date':
                        email_data['date'] = header['value']
                
                body = (plain_text_body or html_body or '') # Use HTML body if available, otherwise use plain text
                email_data['body'] = body # Add the body to the email data for llm parsing

                # Match content
                content_match = False
                if content_keywords:
                    content_to_search = email_data['subject'] + body.lower() # Search in HTML or plain text if HTML is missing
                    # print(content_to_search)
                    for keyword in content_keywords:
                        if keyword.lower() in content_to_search:
                            content_match = True
                            break # If any keyword is found, this email doesn't match
                        if content_match:
                            break

                if content_match:
                     fetched_emails.append(email_data)
        return fetched_emails

    except HttpError as error:
        print(f'An API error occurred: {error}')
        raise error
    except FileNotFoundError as e:
        print("Error: 'credentials.json' not found. Please download it from the Google Cloud Console.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

# Example usage:
# if __name__ == '__main__':
#     # To run this example:
#     # 1. Go to the Google Cloud Console (console.cloud.google.com).
#     # 2. Create a new project or select an existing one.
#     # 3. Enable the Gmail API for your project.
#     # 4. Go to "Credentials" and create OAuth 2.0 Client IDs.
#     # 5. Choose "Desktop app" and create the credentials.
#     # 6. Download the client configuration JSON file and rename it to 'credentials.json'.
#     # 7. Place 'credentials.json' in the same directory as this Python script.
#     # 8. Run the script. It will open a browser window for authentication the first time.

#     # print("Fetching emails from INBOX ...")
#     emails = fetch_emails(labels=['INBOX', 'UPDATES'], start_date='2025/03/09', end_date='2025/03/20',
#                           title_keywords=['Your itinerary', 'booking', 'reservation', 'trip', 'flight', 'hotel', 'emirates', 'airline'],
#                           content_keywords=['emirates', 'trip', 'fly', 'booking', 'hotel', 'itinerary', 'flight', 'reservation', 'airbnb', 'booking'])
#     if emails:
#         for email in emails:
#             print("-" * 20)
#             print(f"ID: {email.get('id')}")
#             print(f"Subject: {email.get('subject')}")
#             print(f"From: {email.get('from')}")
#             print(f"Date: {email.get('date')}")
#     else:
#         print("No emails found matching the criteria.")

    # print("\n" + "=" * 30 + "\n")
