import streamlit as st
import uuid
import openai
from PIL import Image
from io import BytesIO
import pandas as pd
import os
import base64
import io
import concurrent.futures
import requests
import json
from mimetypes import guess_type
import re
import cv2
import numpy as np
from datetime import datetime
import tempfile

# Configure page layout
st.set_page_config(page_title="Insurance Certificate Classifier", page_icon="📜", layout="wide")

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Add OpenAI API settings to session state if not present
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
    st.session_state.endpoint = endpoint
    st.session_state.api_key = api_key
    
# Create folders for processing
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.pdf_folder = os.path.join(st.session_state.temp_dir, "pdfs")
    st.session_state.image_folder = os.path.join(st.session_state.temp_dir, "images")
    os.makedirs(st.session_state.pdf_folder, exist_ok=True)
    os.makedirs(st.session_state.image_folder, exist_ok=True)

# Initialize session state for storing certificates
if 'certificates' not in st.session_state:
    st.session_state.certificates = pd.DataFrame(columns=[
        'Template Form', 'Page Count', 'Name of file',
        'Automobile Liability Insurance Company', 'Automobile Liability Currency',
        'Automobile Liability Amount', 'Automobile Liability DED. Currency',
        'Automobile Liability DED. Amount', 'Automobile Liability Expiry Date (yyyy/mm/dd)',
        'Each occ Commercial General Liability Insurance Company', 'Each occ Commercial General Liability Currency',
        'Each occ Commercial General Liability Amount', 'Each occ Commercial General Liability DED. Currency',
        'Each occ Commercial General Liability DED. Amount', 'Each occ Commercial General Liability Expiry Date (yyyy/mm/dd)',
        'Non-owned Trailer Insurance Company', 'Non-owned Trailer Currency',
        'Non-owned Trailer Amount', 'Non-owned Trailer DED. Currency',
        'Non-owned Trailer DED. Amount', 'Non-owned Trailer Amount Expiry Date (yyyy/mm/dd)',
        'Additional insured', 'Certificate Holder', 'Cancellation Notice Period (days)'
    ])

# Initialize form values in session state
if 'form_values' not in st.session_state:
    st.session_state.form_values = {
        'template_form_value': "",
        'cert_number_value': "",
        'effective_date_value': "",
        'expiration_date_value': "",
        'insured_name_value': "",
        'address_value': "",
        'description_value': "",
        'auto_liability_insurance_company_value': "",
        'auto_liability_currency_value': "",
        'auto_liability_amount_value': "",
        'auto_liability_ded_currency_value': "",
        'auto_liability_ded_amount_value': "",
        'auto_liability_expiry_date_value': "",
        'cgl_company_value': "",
        'cgl_currency_value': "",
        'cgl_amount_value': "",
        'cgl_ded_currency_value': "",
        'cgl_ded_amount_value': "",
        'cgl_expiry_value': "",
        'trailer_company_value': "",
        'trailer_currency_value': "",
        'trailer_amount_value': "",
        'trailer_ded_currency_value': "",
        'trailer_ded_amount_value': "",
        'trailer_expiry_value': "",
        'additional_insured_value': "",
        'certificate_holder_value': "",
        'cancellation_period_value': ""
    }

# Function to export the data as an Excel file
def export_to_excel(dataframe):
    towrite = BytesIO()
    dataframe.to_excel(towrite, index=False, header=True)
    towrite.seek(0)  # reset pointer
    return towrite

# --------------------- Image Preprocessing Functions ---------------------

def convert_pdf_to_images(pdf_path, output_folder):
    """Step 1: PDF → Image"""
    try:
        import pdf2image
        images = pdf2image.convert_from_path(pdf_path)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i + 1}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        return image_paths, len(images)
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return [], 0

def preprocess_image(image):
    """
    Improve image quality for OCR:
      - Convert to OpenCV format and grayscale.
      - Find external contours and crop to the largest contour.
      - Apply fixed threshold then adaptive thresholding.
    """
    # Convert PIL image to OpenCV format (RGB to BGR)
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # Find contours and crop to the largest contour
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnt = cnts_sorted[0]
        x, y, w, h = cv2.boundingRect(cnt)
        gray = gray[y:y+h, x:x+w]
    # Apply fixed threshold and adaptive thresholding
    _, thresh = cv2.threshold(gray, 200, 235, cv2.THRESH_BINARY)
    adaptive_thresh = cv2.adaptiveThreshold(
        thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    processed_image = Image.fromarray(adaptive_thresh)
    return processed_image

def convert_image_to_base64(image_path):
    """Step 2: Image → Base64"""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"

def get_raw_text(image_data_url):
    """
    Step 3a: Extract raw text (OCR) from the image.
    The prompt instructs the LLM to return the plain text found in the image.
    """
    if not st.session_state.api_configured:
        st.error("API credentials not configured.")
        return None
        
    
    # system_prompt = """
    system_prompt = """
    You are an AI assistant designed to tackle complex tasks with the reasoning capabilities of a human genius. Your goal is to complete user-provided tasks while demonstrating thorough self-evaluation, critical thinking, and the ability to navigate ambiguities. You must only provide a final answer when you are 100% certain of its accuracy.

    Here is the task you need to complete:

    <user_task>
    ────────────────────────────────────────────────────────
    1. PURPOSE AND OUTPUT REQUIREMENTS
    ────────────────────────────────────────────────────────
    1.1 Goal: Extract and structure insurance certificate data with absolute accuracy and consistency, adhering to the specified schema. Ensure no assumptions, inferences, or omissions occur, and that all extracted elements follow the required format, especially regarding dates and numeric values.

    Key Objectives:
    • Parse multi-page insurance certificates into a single, coherent JSON output.  
    • Precisely extract only the specified data fields; mark all missing or unclear values as instructed.  
    • Maintain strict adherence to date (yyyy/mm/dd) and numeric formatting rules.  
    • Omit all non-insurance, handwritten, or inferred details.

    1.2 Critical Requirements:
    • Extract ONLY the specified fields
    • Strict JSON format – no deviations
    • Missing fields must be explicitly labeled as "missing"
    • Multi-page documents must be combined into a single structured JSON object
    • No assumptions or inferences: Only extract what is explicitly visible in the PDF. Do not infer, assume, or guess any values not clearly provided. If any data is ambiguous or unclear, mark the field as "[unclear]".
    • ALL dates MUST be in yyyy/mm/dd format WITHOUT EXCEPTION
    • NEVER extract "bodily injury" amounts
    • Currency validation: Follow insured name's address currency
    • Numeric Accuracy: Ensure that numeric values are extracted exactly as they appear. For example, if the PDF shows "10,000," do not change it to "100,000" or vice versa. Remove formatting (such as commas or currency symbols) only after verifying that the underlying digit sequence remains unchanged.
    • Ensure all extracted data is logically consistent

    ────────────────────────────────────────────────────────
    2. VALID EXTRACTION SCOPE
    ────────────────────────────────────────────────────────
    EXTRACT the following details when explicitly present:          
    • Insurance certificate template format
    • Insurance company names
    • Policy amounts and coverage details
    • Deductibles and currency values
    • Policy expiration dates
    • Certificate holder details
    • Additional insured parties
    • Cancellation notice period
            

    DO NOT EXTRACT:
    • Handwritten notes
    • Non-insurance-related text
    • Document metadata outside the specified schema
    • Any inferred data—extract only what is explicitly present
    • Non-owned Automobile
    • "Bodily Injury" (DO NOT extract for Non-Owned Trailer)

    ────────────────────────────────────────────────────────
    3. DATA EXTRACTION AND MAPPING
    ────────────────────────────────────────────────────────
    3.1 Type of Insurance
        • For each row, identify the "Type of Insurance"
        • Extract data only if the row corresponds to a valid insurance type with non-blank values
        • If one type of insurance **CANNOT** be found, automatically mark all fields under that section as "missing". For example, "Automobile Liability" is not found, all fields under it will be marked "missing". 
        • Extract data only if the row corresponds to a valid insurance type with non-blank values. Otherwise, mark all associated fields for that type as "missing".

        3.1.1 Automobile Liability Insurance Handling:
            • Extract its insurance company, coverage amount, deductible, currency, and expiry date
            • ❌ NOT THE SAME AS "Automobile Owners Form"
            • If no clear data is found, label all related fields as "missing"

        3.1.2 Commercial General Liability Insurance Handling:
            • ⚠️ IF EACH OCCURRENCE IS PRESENT, Commercial General Liability AMOUNT refers to EACH OCCURRENCE
            • ⚠️ "Each occurrence" is NOT the same as "General Aggregate". DO NOT CONFUSE.

        3.1.3 Non-Owned Trailer Insurance Handling:
            • ❌ NEVER extract 'Non-Owned Automobile' or 'Trailer Interchange' amounts when extracting Non-Owned Trailer insurance. These are completely separate fields. If 'Non-Owned Automobile' or 'Trailer Interchange' appears, IGNORE IT.
            • ❌ NEVER extract data from "Bodily Injury" for Non-Owned Trailer.
            • If "Non-Owned Trailer" is found anywhere in the document, reference it when extracting.
            • If it is not in a structured row, search alternative fields such as:
                - "Description", "Remarks", "Additional Coverages", "SEF 27", "MSEF 27", "Regarding Insurance Verification" or similar sections.
            • Extract its insurance company, coverage amount, deductible, currency, and expiry date.
            • If no clear data is found, label all related fields as "missing".

    3.2 Insurance Company
        3.2.1 Automobile Liability Insurance Company:
            • Extract company name from the "Automobile Liability" or "Auto Liability" section.
            • Extract only the name, excluding policy numbers.
            • If a "CO LTR" or "Ins." column exists, reference the FULL company name values.
        
        3.2.2 Each occ Commercial General Liability Insurance Company:
            • Extract company name from the "Commercial General Liability" section.
            • Extract only the name, excluding policy numbers.
            • If a "CO LTR" or "Ins." column exists, reference the FULL company name values.
        
        3.2.3 Non-Owned Trailer Insurance Company:
            • Extract company name from the "Non-Owned Trailer" section.
            • Extract only the name, excluding policy numbers.
            • If a "CO LTR" or "Ins." column exists, reference the FULL company name values.
        
        3.2.4 Multiple Insurers Handling:
            • Sometimes, there are multiple insurers (e.g., Insurer A, B, C, etc.).
            • Insurers for each type of insurance are separate; an insurer providing one type does not automatically provide another—check each type individually.
            • Refer to the "Ins." column if present; if multiple insurers are referenced only by letter (A, B, C, etc.) in the subsequent coverage sections, create a list of insurers and map each letter to the corresponding full insurer name.
            • **Detailed Insurer Mapping:** If an insurer is referenced by a letter and no corresponding full name is explicitly found in the document, output the letter reference but flag it as "[unclear]" or log it for manual review to avoid misattribution.
            • Output the name of the insurer, not just the letter, for each insurance type.

    3.3 Amount Extraction
        ⚠️ Extract ONLY the numeric values that are explicitly visible.
        ⚠️ No inference is allowed: Do not adjust digit counts or infer values if the PDF explicitly shows a number. If the visible number is ambiguous, mark the field as "[unclear]".

        3.3.1 Deductible Terminology
            ✅ Valid deductible identifiers (exact matches only):
                • "DED."
                • "DED"
                • "Deductible"
                • "Deductibles"
                • "all perils"
                • "all perils deductible"
                • "Deductible - All Perils"
            
            ❌ Invalid deductible identifiers:
                • Non-Owned Trailer Amount for Bodily Injury (DO NOT extract Bodily Injury for Non-Owned Trailer!!!)
            
        3.3.2 Amount Formatting and Comprehensive Numeric Extraction
            • Extract numeric values only, ensuring complete fidelity to the raw text extracted earlier.
            • Remove any currency symbols, commas, or extraneous formatting only after verifying that the underlying digit sequence remains exactly as shown.
            • Validate that the number’s digit sequence, including decimals if present, is preserved. For example, if the raw text is "10,000", ensure the output is "10000" without altering the digit count.
            • If any discrepancies or formatting anomalies are detected (e.g., misplaced decimals, extra spaces, or unusual separators), flag the field as "[unclear]" and log the anomaly for further review.
            • Implement consistency checks on numeric values to ensure that they match expected formats (e.g., no loss of decimal precision or incorrect digit reordering).

        3.3.3 Field-Specific Rules
            1. Automobile Liability Amount
            ✅ Valid identifiers (exact matches only):
                • Automobile Liability Limits of Liability/Limits
            
            2. Automobile Liability DED. Amount (NOT THE SAME AS NON-OWNED TRAILER AMOUNT)
            ✅ Valid identifiers:
                • Automobile Liability Deductible (refer to valid deductible terminology above)
            ❌ Invalid identifiers (exact matches only):
                • Non-owned Trailer Deductible Amount
                • Each occ Commercial General Liability Deductible Amount
                • Non-Owned Automobile Deductible Amount
            
            3. Each occ Commercial General Liability Amount
            ✅ Valid identifier (exact matches only):
                • Each Occurrence
            ❌ Invalid identifier (exact matches only):
                • General Aggregate
            
            4. Each occ Commercial General Liability DED. Amount
            ✅ Valid identifiers:
                • Each Occurrence Deductible (refer to valid deductible terminology above)
            ❌ Invalid identifiers (exact matches only):
                • Automobile Liability Deductible Amount
                • Non-owned Trailer Deductible Amount
            
            5. Non-owned Trailer Amount
            ✅ Valid identifiers (exact matches only):
                • Non-owned Trailer Limits of Liability/Limits
                • Non-Owned Trailer Physical Damage Limit Per Unit
                • SEF 27
                • MSEF 27
                • Legal liability for damage to non-owned units -- heavy commercial, tractors, and **trailers.**
                • N.O.A. - Trailers
                • Non Owned Trailer Interchange
            ❌ Invalid identifiers (exact matches only):
                • Non-owned Automobile
                • Bodily Injury
                • Automobile Liability
                • SEF23A
                • Medical Expense
                • Trailer Interchange
            
            6. Non-owned Trailer DED. Amount
            ✅ Valid identifiers:
                • Non-owned Trailer Deductible Amount (refer to valid deductible terminology above)
                • N.O.A. - Trailers Deductible Amount
                • Non Owned Trailer Interchange Deductible Amount
            ❌ Invalid identifiers (exact matches only):
                • Automobile Liability Deductible Amount
                • Each occ Commercial General Liability Deductible Amount
                • SEF23A (If SEF23A is found, COMPREHENSIVELY IGNORE ALL VALUES RELATED TO IT)
                • Trailer Interchange Deductible Amount
            
        3.3.4 Special Rules for Non-owned Trailer Amount
            1. Search the entire document for the amount.
            2. Valid locations:
                • Dedicated Non-owned Trailer section
                • Policy Number section (in line with Non-owned Trailer amount)
                • Any section with a clear indication of Non-owned Trailer coverage
            3. Exclusions:
                ❌ UNDER ALL CIRCUMSTANCES, DO NOT EXTRACT "BODILY INJURY" AT ANY TIME.
                • If multiple amounts exist for Non-Owned Trailer, use the NON-bodily-injury amount.
                • If there are two separate Non-Owned Trailer coverages, DO NOT EXTRACT the one that mentions bodily injury.
        
        3.3.5 Handling Other Liability:
            • In cases where multiple values are present in a single cell (e.g., for other liability), utilize alignment cues and spatial separation (e.g., column delimiters) to correctly separate and extract these values.
            • Reinforce that the insurance deductible is separate for each insurance type; the AI must search on a row-by-row basis. If there is no value for a particular row (for either the coverage amount or the deductible), that field must be tagged as "missing".

    3.4 Currency Assignment
        Determine Currency Based on Insured’s Address
        • If the address explicitly indicates a Canadian location (e.g., “Alberta, Canada” or a valid Canadian postal code), set currency to "CAD".
        • If the address explicitly indicates a United States location (e.g., “Seattle, WA” or a valid U.S. ZIP code), set currency to "USD".
        • If the address clearly belongs to another country and the currency is explicitly stated (e.g., “GBP” or “EUR” for a UK/EU address), use that currency.
        • If the address is ambiguous or does not clearly show a specific country, do not infer currency—mark all currency fields as "missing".

        Link Currency to Each Coverage or Deductible
        • For any coverage or deductible you successfully extract, assign the currency determined above.
        • If you cannot extract a coverage or deductible value for a particular insurance type (i.e., it is “missing”), then the corresponding currency field must also be "missing".

        No Assumptions, No Inferences
        • Do not guess currency based on any indirect information (e.g., phone numbers or area codes). If the document does not explicitly confirm Canada, the U.S., or another recognized currency, label the currency as "missing".
        • If any data is unclear or contradictory, mark it as "[unclear]".

    3.5 Date Parsing and Standardization
        ⚠️ All extracted dates MUST be converted to yyyy/mm/dd format

            
        3.5.1 Date Format Rules for Initial Extraction
            CRITICAL: When you see dates in YY/MM/DD format (like 24/11/15):
            • The FIRST number is ALWAYS the YEAR
            • The SECOND number is ALWAYS the MONTH
            • The THIRD number is ALWAYS the DAY
            
            Examples of correct initial extraction:
            • 24/11/15 should be extracted as 2024/11/15 (NOT 11/24/2015)
            • 23/05/20 should be extracted as 2023/05/20 (NOT 05/23/2020)
            • 25/12/31 should be extracted as 2025/12/31 (NOT 12/25/2031)
            
            ❌ INCORRECT interpretations:
            • 24/11/15 as November 24, 2015
            • 23/05/20 as May 23, 2020
            • 25/12/31 as December 25, 2031
            
            CRITICAL: When you see dates in MM/DD/YYYY format (like 11/03/2024):
            • The FIRST number is ALWAYS the MONTH
            • The SECOND number is ALWAYS the DATE
            • The THIRD number is ALWAYS the YEAR
            
            Examples of correct initial extraction:
            • 11/03/2024 should be extracted as 2024/11/03 (NOT 03/11/2024)
            
            ❌ INCORRECT interpretations:
            • 11/03/2024 as March 11, 2024

    3.6 Additional Fields
        3.6.1 Certificate Holder
            • Look for fields labeled specifically as "Certificate Holder"
            • Also look for text preceded by phrases like "This is to certify to..."
            • If an address is present, include it. If no clear data is found, label as "missing"
            • Common values are: "C. Keay Investments Ltd. DBA Ocean Trailer, 9076 River Road Delta, BC V4G 1B5", "To Whom it May Concern", "Keay Investments LTD o/a Ocean Trailer, 234136 84 ST SE, Rocky View, AB T1X 0K2". Otherwise, extract what is present.
            • Sometimes written as "To Whom it May Concern", EXTRACT AS IT IS.
            • Common locations:
                - In a dedicated box or field usually in the bottom portion
                - In a section following "This certificate is issued to..."
                - In a section starting with "This is to certify to..."
        
        3.6.2 Additional Insured
            • Labeled ONLY as "Additional Insured". NO OTHER LABELS.
            • ❌ NOT THE SAME AS "CERTIFICATE HOLDER"
            • ❌ NOT THE SAME AS "ADDITIONAL INFORMATION"
            • Include the listed company name along with their addresses if present.
            • Common values are: "Certificate Holder but only with respect to work performed under contract by the Named Insured (CGL Only)", "C. Keay Investments Ltd. DBA Ocean Trailer, 9076 River Road Delta, BC V4G 1B5", "Keay Investments LTD o/a Ocean Trailer, 234136 84 ST SE, Rocky View, AB T1X 0K2". Otherwise, extract what is present.
            • ALWAYS extract the sentence immediately following the "Additional Insured" field.
            • Common locations:
                - In a dedicated "Additional Insured" section.
            • If no clear data is found, label as "missing".

    3.7 Handling Missing and Unclear Data
        • Explicitly mark missing values as "missing".
        • If a field is unclear across pages, flag it as "[unclear]".
        • No assumptions—only extract what is present.

    ────────────────────────────────────────────────────────
    4. OUTPUT FORMAT
    ────────────────────────────────────────────────────────
    Produce for each parsed document a single JSON object containing exactly the following fields:

    {
            "Template Form": "[Monarch|Lloyd Sadd|NFP|CSIO|Rogers|Wylie Crump|MHK|Fleet|ACORD|O HUB|All Insurance Ltd.|WESTLAND|Mango Insurance|AON|Goldkey Insurance|Brokerlink|One Insurance|A-KAN|Ing+Mckee|BFL Canada Insurance Services Inc.|Co-Operators|Federated Insurance|Prl|Foster Park|Risktech Insurance Services Inc.|Drayden Insurance|Unknown]",
            "Page Count": "integer",
            "Automobile Liability Insurance Company": "string",
            "Automobile Liability Currency": "string",
            "Automobile Liability Amount": "[integer|string]",
            "Automobile Liability DED. Currency": "string",
            "Automobile Liability DED. Amount": "[integer|string]",                          // NOT THE SAME AS 'NON-OWNED TRAILER DED. AMOUNT'
            "Automobile Liability Expiry Date (yyyy/mm/dd)": "date",                         // ALWAYS REFER TO Date Parsing and Standardization. STRICT FORMAT: yyyy/mm/dd ONLY
            "Each occ Commercial General Liability Insurance Company": "string",
            "Each occ Commercial General Liability Currency": "string",
            "Each occ Commercial General Liability Amount": "[integer|string]",
            "Each occ Commercial General Liability DED. Currency": "string",
            "Each occ Commercial General Liability DED. Amount": "[integer|string]",
            "Each occ Commercial General Liability Expiry Date (yyyy/mm/dd)": "date",        // ALWAYS REFER TO Date Parsing and Standardization. STRICT FORMAT: yyyy/mm/dd ONLY
            "Non-owned Trailer Insurance Company": "string",
            "Non-owned Trailer Currency": "string",
            "Non-owned Trailer Amount": "[integer|string]",
            "Non-owned Trailer DED. Currency": "string",
            "Non-owned Trailer DED. Amount": "[integer|string]",
            "Non-owned Trailer Amount Expiry Date (yyyy/mm/dd)": "date",                     // ALWAYS REFER TO Date Parsing and Standardization. STRICT FORMAT: yyyy/mm/dd ONLY
            "Additional insured": "string",
            "Certificate Holder": "string",
            "Cancellation Notice Period (days)": "[integer|string]"
        }

    Notes:  
    • Mark any field explicitly absent as “missing.”  
    • If the data is visible but unclear, use “[unclear].”  
    • Ensure no extraneous keys are added.  
    • Numeric accuracy: preserve the exact digit sequence shown, removing commas or currency symbols only after verifying correctness.

    ────────────────────────────────────────────────────────
    5. OUTPUT EXAMPLE
    ────────────────────────────────────────────────────────
    Below is a sample JSON showing correct formatting and data categorization (for demonstration; actual extracted values will vary):

    {
            "Template Form": "CSIO",
            "Page Count": "2",
            "Automobile Liability Insurance Company": "ABC Insurance Co.",
            "Automobile Liability Currency": "CAD",
            "Automobile Liability Amount": "500,000.00",
            "Automobile Liability DED. Currency": "CAD",
            "Automobile Liability DED. Amount": "1,000.00",
            "Automobile Liability Expiry Date (yyyy/mm/dd)": "2025/01/31",                         // ALWAYS REFER TO Date Parsing and Standardization. STRICT FORMAT: yyyy/mm/dd ONLY
            "Each occ Commercial General Liability Insurance Company": "XYZ Insurance",
            "Each occ Commercial General Liability Currency": "USD",
            "Each occ Commercial General Liability Amount": "1,000,000.00",
            "Each occ Commercial General Liability DED. Currency": "USD",
            "Each occ Commercial General Liability DED. Amount": "50,000.00",
            "Each occ Commercial General Liability Expiry Date (yyyy/mm/dd)": "2026/01/15",        // ALWAYS REFER TO Date Parsing and Standardization. STRICT FORMAT: yyyy/mm/dd ONLY
            "Non-owned Trailer Insurance Company": "DEF Insurance",
            "Non-owned Trailer Currency": "CAD",
            "Non-owned Trailer Amount": "5,000.00",
            "Non-owned Trailer DED. Currency": "CAD",
            "Non-owned Trailer DED. Amount": "750.00",
            "Non-owned Trailer Amount Expiry Date (yyyy/mm/dd)": "2026/06/30",                     // ALWAYS REFER TO Date Parsing and Standardization. STRICT FORMAT: yyyy/mm/dd ONLY
            "Additional insured": "missing",
            "Certificate Holder": "Company XYZ 123 Main Street, Toronto, ON M5J 2N8, Canada",
            "Cancellation Notice Period (days)": "30"
        }

    ────────────────────────────────────────────────────────
    6. IMPORTANT NOTES
    ────────────────────────────────────────────────────────
    • Numeric Integrity: Under no circumstance alter the digit count. For instance, “10,000” should become “10000,” preserving the “10000” sequence precisely.  
    • Deductible isolation: Avoid mixing coverage amounts with deductibles or referencing the wrong insurance type. Match each deductible to its coverage type.  
    • Non-Owned Trailer Coverage: Strictly exclude bodily injury amounts and “Non-Owned Automobile.” Only capture trailer-specific coverage.  
    • If multiple insurers appear, map them carefully (e.g., from columns labeled “Insurer A,” “Insurer B,” etc.). Do not guess the full name if only letters are provided—use “[unclear]” for unresolvable insurer names.

    ────────────────────────────────────────────────────────
    7. FINAL REMINDERS
    ────────────────────────────────────────────────────────
    • Multi-page documents must be combined into a single structured output.
    • Ensure Page Count reflects the number of pages processed for each document.
    • No assumptions - extract only what is explicitly visible; do not infer missing values.
    • Ensure structure uniformity—every response must match the specified JSON format.
    • Clearly mark missing fields using "missing" instead of leaving fields blank.
    • Strict Numeric Integrity: Preserve numeric values exactly as they appear in the PDF. Any transformation must not alter the actual digits, their count, or their order.
    • The AI must follow all insurance handling rules and extraction logic as explicitly mentioned in this prompt, including handling each insurance type’s deductible and coverage on a row-by-row basis, separate insurer mapping per insurance type with detailed handling for letter references, and the hard coded logic for overriding corresponding currency fields when amounts are missing.

    </user_task>

    Please follow these steps carefully:

    1. Initial Attempt:

    Make an initial attempt at completing the task. Present this attempt in <initial_attempt> tags.

    2. Self-Evaluation:

    Critically evaluate your initial attempt. Identify any areas where you are not completely certain or where ambiguities exist. List these uncertainties in <doubts> tags.

    3. Self-Prompting:

    For each doubt or uncertainty, create self-prompts to address and clarify these issues. Document this process in <self_prompts> tags.

    4. Chain of Thought Reasoning:

    Wrap your reasoning process in <reasoning> tags. Within these tags, use the following structure to organize your thoughts:

    # Key Information
    # Task Decomposition
    # Structured Plan
    # Analysis and Multiple Perspectives
    # Assumptions and Biases
    # Alternative Approaches
    # Risks and Edge Cases
    # Testing and Revising
    # Metacognition and Self-Analysis
    # Strategize and Evaluate
    # Backtracking and Correcting

    In each section, do the following:
    • Explain what information you have extracted from the images.  
    • Break the parsing task into smaller pieces.  
    • Develop a structured plan for extracting each data item.  
    • Analyze different possibilities for ambiguous fields or partial data.  
    • Discuss potential biases or pitfalls in your approach.  
    • Explore alternative strategies if needed.  
    • Test your partial solutions and refine them as necessary.  
    • Reflect on your own thought process and adjust strategy where needed.  

    5. Uncertainty Check:

    After your thorough analysis, assess whether you can proceed with 100% certainty. If you still have unresolved ambiguities, clearly state that you cannot provide a final answer and explain why in <failure_explanation> tags.

    6. Final Answer:

    Only if you are absolutely certain of your conclusion, present your final answer in <answer> tags. Your answer must strictly be valid JSON that follows the "OUTPUT STRUCTURE" from <user_task>. Explain why you are confident in this solution.
"""
    
    try:
        with st.spinner("Processing OCR..."):
            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data_url}}]}
                ],
                "max_tokens": 2000,
                "temperature": 0
            }
            
            headers = {
                "Content-Type": "application/json",
                "api-key": st.session_state.api_key
            }
            
            response = requests.post(st.session_state.endpoint, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                st.error(f"OCR API response code: {response.status_code}")
                st.error(response.text)
            return None
    except Exception as e:
        st.error(f"OCR API error: {str(e)}")
        return None

def parse_structured_response(response_content):
    """Robustly extract structured JSON from LLM response."""
    # If response_content is already a dict, return it directly
    if isinstance(response_content, dict):
        return response_content

    # If response_content is a string, extract JSON from <initial_attempt> tags
    if isinstance(response_content, str):
        json_match = re.search(r'<initial_attempt>\s*```json(.*?)```\s*</initial_attempt>', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                structured_data = json.loads(json_str)
                return structured_data
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {e}")
                st.error(f"Extracted JSON was: {json_str}")
                return None
        else:
            # Try to find any JSON block in the response as a fallback
            json_block = re.search(r'```json(.*?)```', response_content, re.DOTALL)
            if json_block:
                json_str = json_block.group(1).strip()
                try:
                    structured_data = json.loads(json_str)
                    return structured_data
                except json.JSONDecodeError:
                    pass
            
            st.error("No valid JSON found in the response.")
            return None

    st.error(f"Unexpected response content type: {type(response_content)}")
    return None

def get_structured_data_from_text(raw_text):
    """Extract structured JSON data from the raw OCR text."""
    if not st.session_state.api_configured:
        st.error("API credentials not configured.")
        return None
        
    # Define the schema expected from the LLM
    schema = {
        "certificateInfo": {
            "certificateNumber": "string",
            "templateForm": "string",
            "effectiveDate": "date (yyyy/mm/dd)",
            "expirationDate": "date (yyyy/mm/dd)",
            "insuredName": "string",
            "address": "string",
            "description": "string"
        },
        "automobileLiability": {
            "insuranceCompany": "string",
            "currency": "string",
            "amount": "number",
            "deductibleCurrency": "string",
            "deductibleAmount": "number",
            "expiryDate": "date (yyyy/mm/dd)"
        },
        "commercialGeneralLiability": {
            "insuranceCompany": "string",
            "currency": "string",
            "amount": "number",
            "deductibleCurrency": "string",
            "deductibleAmount": "number",
            "expiryDate": "date (yyyy/mm/dd)"
        },
        "nonOwnedTrailer": {
            "insuranceCompany": "string",
            "currency": "string",
            "amount": "number",
            "deductibleCurrency": "string",
            "deductibleAmount": "number",
            "expiryDate": "date (yyyy/mm/dd)"
        },
        "other": {
            "additionalInsured": "string",
            "certificateHolder": "string",
            "cancellationNoticePeriod": "number (days)"
        }
    }
    
    # Create a clear system prompt that provides the expected schema
    system_prompt = f"""
    You are an AI assistant specialized in extracting insurance certificate data.
    
    Extract data from the provided insurance certificate text according to this schema:
    {json.dumps(schema, indent=2)}
    
    Follow these rules:
    1. Extract all available information that fits the schema.
    2. If information is missing, leave the field empty.
    3. For currency fields, use the standard 3-letter currency code (e.g., USD, EUR).
    4. For date fields, use the format yyyy/mm/dd.
    5. For amount fields, extract only the numeric value without currency symbols or commas.
    6. If multiple values could fit a field, choose the most appropriate one.
    7. Respond ONLY with a valid JSON object following the schema.
    """
    
    try:
        with st.spinner("Extracting structured data..."):
            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract and structure the information based on the following extracted text. Provide your response strictly in JSON format wrapped within ```json and ``` inside <initial_attempt> tags."},
                        {"type": "text", "text": raw_text}
                    ]}
                ],
                "max_tokens": 2000,
                "temperature": 0.2  # Lower temperature for more consistent extraction
            }
            
            headers = {
                "Content-Type": "application/json",
                "api-key": st.session_state.api_key
            }
            
            response = requests.post(st.session_state.endpoint, headers=headers, json=data)
            response.raise_for_status()
            response_content = response.json()["choices"][0]["message"]["content"]

            # Use the robust parsing function
            structured_data = parse_structured_response(response_content)
            
            # Log the structured data for debugging
            st.session_state.last_structured_data = structured_data
            return structured_data

    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.error(traceback.format_exc())

    return None

def process_document(file_content, file_name):
    """Process a document (image or PDF) and extract information"""
    # Initialize OCR if not already done
    if not hasattr(st.session_state, 'ocr_processor'):
        st.session_state.ocr_processor = initialize_ocr()
    
    try:
        # Extract text using OCR - this would be replaced with your actual OCR implementation
        with st.spinner("Extracting text from document..."):
            if file_name.lower().endswith('.pdf'):
                # Process PDF file
                raw_text = extract_text_from_pdf(file_content)
            else:
                # Process image file
                raw_text = extract_text_from_image(file_content)
            
            if not raw_text:
                st.error("No text could be extracted from the document.")
                return None
            
            # Show extracted text in an expander for debugging
            with st.expander("View Extracted Text"):
                st.text(raw_text)
            
            # Get structured data from the raw text
            structured_data = get_structured_data_from_text(raw_text)
            
            if structured_data:
                # Convert the nested structure to flat dictionary for form values
                flat_data = flatten_structured_data(structured_data)
                
                # Update the form values
                update_form_values(flat_data)
                
                return structured_data
            else:
                st.error("Failed to extract structured data from the document.")
                return None
                
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.error(traceback.format_exc())
        return None

def flatten_structured_data(structured_data):
    """Convert nested structured data to a flat dictionary for form values."""
    flat_data = {}
    
    # Certificate Info
    if "certificateInfo" in structured_data:
        info = structured_data["certificateInfo"]
        flat_data["cert_number_value"] = info.get("certificateNumber", "")
        flat_data["template_form_value"] = info.get("templateForm", "")
        flat_data["effective_date_value"] = info.get("effectiveDate", "")
        flat_data["expiration_date_value"] = info.get("expirationDate", "")
        flat_data["insured_name_value"] = info.get("insuredName", "")
        flat_data["address_value"] = info.get("address", "")
        flat_data["description_value"] = info.get("description", "")
    
    # Automobile Liability
    if "automobileLiability" in structured_data:
        auto = structured_data["automobileLiability"]
        flat_data["auto_liability_insurance_company_value"] = auto.get("insuranceCompany", "")
        flat_data["auto_liability_currency_value"] = auto.get("currency", "")
        flat_data["auto_liability_amount_value"] = auto.get("amount", "")
        flat_data["auto_liability_ded_currency_value"] = auto.get("deductibleCurrency", "")
        flat_data["auto_liability_ded_amount_value"] = auto.get("deductibleAmount", "")
        flat_data["auto_liability_expiry_date_value"] = auto.get("expiryDate", "")
    
    # Commercial General Liability
    if "commercialGeneralLiability" in structured_data:
        cgl = structured_data["commercialGeneralLiability"]
        flat_data["cgl_company_value"] = cgl.get("insuranceCompany", "")
        flat_data["cgl_currency_value"] = cgl.get("currency", "")
        flat_data["cgl_amount_value"] = cgl.get("amount", "")
        flat_data["cgl_ded_currency_value"] = cgl.get("deductibleCurrency", "")
        flat_data["cgl_ded_amount_value"] = cgl.get("deductibleAmount", "")
        flat_data["cgl_expiry_value"] = cgl.get("expiryDate", "")
    
    # Non-owned Trailer
    if "nonOwnedTrailer" in structured_data:
        trailer = structured_data["nonOwnedTrailer"]
        flat_data["trailer_company_value"] = trailer.get("insuranceCompany", "")
        flat_data["trailer_currency_value"] = trailer.get("currency", "")
        flat_data["trailer_amount_value"] = trailer.get("amount", "")
        flat_data["trailer_ded_currency_value"] = trailer.get("deductibleCurrency", "")
        flat_data["trailer_ded_amount_value"] = trailer.get("deductibleAmount", "")
        flat_data["trailer_expiry_value"] = trailer.get("expiryDate", "")
    
    # Other fields
    if "other" in structured_data:
        other = structured_data["other"]
        flat_data["additional_insured_value"] = other.get("additionalInsured", "")
        flat_data["certificate_holder_value"] = other.get("certificateHolder", "")
        flat_data["cancellation_period_value"] = other.get("cancellationNoticePeriod", "")
    
    return flat_data

def update_form_values(flat_data):
    """Update form values from processed data."""
    for key, value in flat_data.items():
        if key in st.session_state.form_values:
            if isinstance(value, (int, float)):
                st.session_state.form_values[key] = str(value)
            else:
                st.session_state.form_values[key] = value

def extract_text_from_pdf(pdf_content):
    """Extract text from a PDF file using PyPDF2."""
    # This is a placeholder - implement with your preferred PDF extraction library
    try:
        from io import BytesIO
        import PyPDF2
        
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_image(image_content):
    """Extract text from an image using OCR."""
    # This is a placeholder - implement with your preferred OCR library
    try:
        import pytesseract
        from PIL import Image
        from io import BytesIO
        
        image = Image.open(BytesIO(image_content))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

def initialize_ocr():
    """Initialize OCR engine."""
    # Placeholder for OCR initialization
    return "OCR initialized"

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'api_configured' not in st.session_state:
        st.session_state.api_configured = False
    
    if 'endpoint' not in st.session_state:
        st.session_state.endpoint = ""
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    if 'certificates' not in st.session_state:
        st.session_state.certificates = pd.DataFrame()
    
    if 'form_values' not in st.session_state:
        st.session_state.form_values = {
            # Certificate Info
            "cert_number_value": "",
            "template_form_value": "",
            "effective_date_value": "",
            "expiration_date_value": "",
            "insured_name_value": "",
            "address_value": "",
            "description_value": "",
            
            # Automobile Liability
            "auto_liability_insurance_company_value": "",
            "auto_liability_currency_value": "",
            "auto_liability_amount_value": "",
            "auto_liability_ded_currency_value": "",
            "auto_liability_ded_amount_value": "",
            "auto_liability_expiry_date_value": "",
            
            # Commercial General Liability
            "cgl_company_value": "",
            "cgl_currency_value": "",
            "cgl_amount_value": "",
            "cgl_ded_currency_value": "",
            "cgl_ded_amount_value": "",
            "cgl_expiry_value": "",
            
            # Non-owned Trailer
            "trailer_company_value": "",
            "trailer_currency_value": "",
            "trailer_amount_value": "",
            "trailer_ded_currency_value": "",
            "trailer_ded_amount_value": "",
            "trailer_expiry_value": "",
            
            # Others
            "additional_insured_value": "",
            "certificate_holder_value": "",
            "cancellation_period_value": ""
        }
    
    if 'last_structured_data' not in st.session_state:
        st.session_state.last_structured_data = None

def export_to_excel(df):
    """Export dataframe to Excel format."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Certificates', index=False)
    output.seek(0)
    return output.getvalue()

# Main function for the Streamlit app
def main():
    st.title("📜 Insurance Certificate Classifier")
    
    # Initialize session state
    initialize_session_state()
    
    st.markdown("Hey there! I'm your new insurance certificate sidekick. Just send me a photo or upload your document, and I'll handle the data entry.")
    
    # Create tabs for API configuration and main app
    tab1, tab2 = st.tabs(["App", "API Settings"])
    
    with tab2:
        st.subheader("API Configuration")
        st.markdown("Enter your Azure OpenAI API credentials below:")
        
        endpoint = st.text_input("API Endpoint URL", value=st.session_state.endpoint)
        api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
        
        if st.button("Save API Settings"):
            st.session_state.endpoint = endpoint
            st.session_state.api_key = api_key
            st.session_state.api_configured = True if endpoint and api_key else False
            if st.session_state.api_configured:
                st.success("API settings saved successfully!")
            else:
                st.error("Please provide both API endpoint and key.")
    
    with tab1:
        # Create a two-column layout with both input options on the left
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            # First option: Take a photo
            st.markdown("<h3 style='color: navy;'>Scan using camera</h3>", unsafe_allow_html=True)
            
            # Apply custom CSS to make the camera input smaller
            st.markdown("""
            <style>
            .element-container:has(.stCamera) {
                max-width: 400px;
                margin: 0 auto;
            }
            .stCamera > div {
                min-height: 300px !important;
                max-height: 300px !important;
            }
            </style> Use your camera to take a clear scan of the document. 
            """, unsafe_allow_html=True)
            
            # Create a form for the upload/camera input
            with st.form(key="upload_form"):
                camera_image = st.camera_input(" ", key="camera")
                
                st.markdown("<div style='text-align: center; font-weight: bold; margin: 20px 0;'>OR</div>", unsafe_allow_html=True)
                
                # Second option: Upload receipt
                st.markdown("<h3 style='color: navy;'>Upload Document</h3>", unsafe_allow_html=True)
                uploaded_files = st.file_uploader(
                    "Drag and drop files here", 
                    accept_multiple_files=True, 
                    type=["pdf", "jpg", "jpeg", "png"],
                    key="uploader"
                )
                st.caption("Limit 200MB per file • PDF, JPG, JPEG, PNG")
                
                # Add the submit button at the end of the form
                submit_button = st.form_submit_button(
                    label="Process Document", 
                    type="primary",
                    icon="✅",
                    use_container_width=True
                )
                
                # Handle the form submission
                if submit_button:
                    if uploaded_files or camera_image:
                        st.success("Analyzing...")
                        
                        # Process camera image if available
                        if camera_image:
                            with st.spinner("Processing camera image..."):
                                processed_data = process_document(camera_image.getvalue(), "camera_image.jpg")
                                if processed_data:
                                    st.success("Camera image processed!")
                        
                        # Process uploaded files if available
                        if uploaded_files:
                            for uploaded_file in uploaded_files:
                                with st.spinner(f"Processing {uploaded_file.name}..."):
                                    processed_data = process_document(uploaded_file.getvalue(), uploaded_file.name)
                                    if processed_data:
                                        st.success(f"{uploaded_file.name} processed!")
                    else:
                        st.error("Oops! Please upload a document or scan first.")
        
        with right_col:
            # Display status cards at the top
            col1, col2 = st.columns(2)
            
            # Calculate verification metrics if structured data exists
            verified_coverages = 0
            total_coverages = 0
            compliance_issues = 0
            
            if st.session_state.last_structured_data:
                data = st.session_state.last_structured_data
                
                # Count verified coverages
                for category in ['automobileLiability', 'commercialGeneralLiability', 'nonOwnedTrailer']:
                    if category in data and any(data[category].values()):
                        verified_coverages += 1
                        total_coverages += 1
                    elif category in data:
                        total_coverages += 1
                
                # Check for compliance issues (expiry dates in the past)
                from datetime import datetime
                today = datetime.now().strftime('%Y/%m/%d')
                
                for category in ['automobileLiability', 'commercialGeneralLiability', 'nonOwnedTrailer']:
                    if category in data and 'expiryDate' in data[category]:
                        expiry = data[category]['expiryDate']
                        if expiry and expiry < today:
                            compliance_issues += 1
            
            with col1:
                st.markdown(f"""
                <div style="background-color:#E8F0FE; border-radius:10px; padding:10px; margin-bottom:20px;">
                    <div style="display:flex; align-items:center;">
                        <div style="color:#4285F4; margin-right:20px;"><i class="fas fa-check-circle"></i>✓</div>
                        <div>
                            <div style="color:#4285F4; font-weight:bold;">AI Extraction Status:</div>
                            <div style="color:#4285F4;">{verified_coverages}/{total_coverages} verified coverages</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color:#FEEAE6; border-radius:10px; padding:10px; margin-bottom:20px;">
                    <div style="display:flex; align-items:center;">
                        <div style="color:#EA4335; margin-right:20px;"><i class="fas fa-exclamation-triangle"></i>⚠</div>
                        <div>
                            <div style="color:#EA4335; font-weight:bold;">Compliance Status:</div>
                            <div style="color:#EA4335;">{compliance_issues} compliance issues</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create a container with scrollable height for the form
            cert_form_container = st.container()

            with cert_form_container:
                with st.form(key="certificate_form"):
                    # Add single scrollable container
                    st.markdown("""
                    <style>
                    .scrollable-form {
                        max-height: 500px;
                        overflow-y: auto;
                        padding-right: 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs
                    tabs = st.tabs([
                        "Certificate Info",
                        "Automobile Liability",
                        "Commercial General Liability",
                        "Non-owned Trailer", 
                        "Others"
                    ])

                    # Tab 1: Certificate & Insured Info
                    with tabs[0]:
                        st.markdown("##### Certificate Details")
                        cert_number = st.text_input("Certificate Number", key="cert_number", 
                                                   value=st.session_state.form_values["cert_number_value"])
                        form_type = st.text_input("Template Form", key="template_form", 
                                                 value=st.session_state.form_values["template_form_value"])
                        
                        col_eff, col_exp = st.columns(2)
                        with col_eff:
                            effective_date = st.text_input("Effective Date (yyyy/mm/dd)", key="effective_date", 
                                                         value=st.session_state.form_values["effective_date_value"])
                        with col_exp:
                            expiration_date = st.text_input("Expiration Date (yyyy/mm/dd)", key="expiration_date", 
                                                          value=st.session_state.form_values["expiration_date_value"])
                    
                        st.markdown("##### Insured Information")
                        insured_name = st.text_input("Insured Name", key="insured_name", 
                                                   value=st.session_state.form_values["insured_name_value"])
                        address = st.text_input("Address", key="address", 
                                              value=st.session_state.form_values["address_value"])
                        description = st.text_area("Description", key="description", 
                                                 value=st.session_state.form_values["description_value"])

                    # Tab 2: Automobile Liability
                    with tabs[1]:
                        auto_liability_company = st.text_input("Automobile Liability Insurance Company", key="auto_liability_company",
                                                            value=st.session_state.form_values["auto_liability_insurance_company_value"])
                        auto_liability_currency = st.text_input("Automobile Liability Currency", key="auto_liability_currency",
                                                             value=st.session_state.form_values["auto_liability_currency_value"])
                        auto_liability_amount = st.text_input("Automobile Liability Amount", key="auto_liability_amount",
                                                           value=st.session_state.form_values["auto_liability_amount_value"])
                        auto_liability_ded_currency = st.text_input("Automobile Liability DED. Currency", key="auto_liability_ded_currency",
                                                                value=st.session_state.form_values["auto_liability_ded_currency_value"])
                        auto_liability_ded_amount = st.text_input("Automobile Liability DED. Amount", key="auto_liability_ded_amount",
                                                              value=st.session_state.form_values["auto_liability_ded_amount_value"])
                        auto_liability_expiry = st.text_input("Automobile Liability Expiry Date (yyyy/mm/dd)", key="auto_liability_expiry",
                                                           value=st.session_state.form_values["auto_liability_expiry_date_value"])
                    
                    # Tab 3: Commercial General Liability
                    with tabs[2]:
                        cgl_company = st.text_input("Each occ Commercial General Liability Insurance Company", key="cgl_company",
                                                 value=st.session_state.form_values["cgl_company_value"])
                        cgl_currency = st.text_input("Each occ Commercial General Liability Currency", key="cgl_currency",
                                                  value=st.session_state.form_values["cgl_currency_value"])
                        cgl_amount = st.text_input("Each occ Commercial General Liability Amount", key="cgl_amount",
                                                value=st.session_state.form_values["cgl_amount_value"])
                        cgl_ded_currency = st.text_input("Each occ Commercial General Liability DED. Currency", key="cgl_ded_currency",
                                                      value=st.session_state.form_values["cgl_ded_currency_value"])
                        cgl_ded_amount = st.text_input("Each occ Commercial General Liability DED. Amount", key="cgl_ded_amount",
                                                    value=st.session_state.form_values["cgl_ded_amount_value"])
                        cgl_expiry = st.text_input("Each occ Commercial General Liability Expiry Date (yyyy/mm/dd)", key="cgl_expiry",
                                                value=st.session_state.form_values["cgl_expiry_value"])
                    
                    # Tab 4: Non-owned Trailer
                    with tabs[3]:
                        trailer_company = st.text_input("Non-owned Trailer Insurance Company", key="trailer_company",
                                                     value=st.session_state.form_values["trailer_company_value"])
                        trailer_currency = st.text_input("Non-owned Trailer Currency", key="trailer_currency",
                                                      value=st.session_state.form_values["trailer_currency_value"])
                        trailer_amount = st.text_input("Non-owned Trailer Amount", key="trailer_amount",
                                                    value=st.session_state.form_values["trailer_amount_value"])
                        trailer_ded_currency = st.text_input("Non-owned Trailer DED. Currency", key="trailer_ded_currency",
                                                         value=st.session_state.form_values["trailer_ded_currency_value"])
                        trailer_ded_amount = st.text_input("Non-owned Trailer DED. Amount", key="trailer_ded_amount",
                                                       value=st.session_state.form_values["trailer_ded_amount_value"])
                        trailer_expiry = st.text_input("Non-owned Trailer Amount Expiry Date (yyyy/mm/dd)", key="trailer_expiry",
                                                    value=st.session_state.form_values["trailer_expiry_value"])
                    
                    # Tab 5: Others
                    with tabs[4]:
                        st.markdown("##### Additional insured")
                        additional_insured = st.text_area("Additional Insured", key="additional_insured",
                                                      value=st.session_state.form_values["additional_insured_value"])
                    
                        st.markdown("##### Certificate Holder")
                        certificate_holder = st.text_area("Certificate Holder", key="certificate_holder",
                                                       value=st.session_state.form_values["certificate_holder_value"])
                    
                        st.markdown("##### Cancellation Notice Period (days)")
                        cancellation_period = st.text_input("Cancellation Notice Period (days)", key="cancellation_period",
                                                         value=st.session_state.form_values["cancellation_period_value"])
                    
                    # Process button
                    process_button = st.form_submit_button("Save Certificate")
                    
                    if process_button:
                        # Process the data
                        certificate_data = {
                            "Template Form": form_type,
                            "Page Count": "1",  # This would be determined by the actual processing
                            "Name of file": "Manually entered",
                            "Automobile Liability Insurance Company": auto_liability_company,
                            "Automobile Liability Currency": auto_liability_currency,
                            "Automobile Liability Amount": auto_liability_amount,
                            "Automobile Liability DED. Currency": auto_liability_ded_currency,
                            "Automobile Liability DED. Amount": auto_liability_ded_amount,
                            "Automobile Liability Expiry Date (yyyy/mm/dd)": auto_liability_expiry,
                            "Each occ Commercial General Liability Insurance Company": cgl_company,
                            "Each occ Commercial General Liability Currency": cgl_currency,
                            "Each occ Commercial General Liability Amount": cgl_amount,
                            "Each occ Commercial General Liability DED. Currency": cgl_ded_currency,
                            "Each occ Commercial General Liability DED. Amount": cgl_ded_amount,
                            "Each occ Commercial General Liability Expiry Date (yyyy/mm/dd)": cgl_expiry,
                            "Non-owned Trailer Insurance Company": trailer_company,
                            "Non-owned Trailer Currency": trailer_currency,
                            "Non-owned Trailer Amount": trailer_amount,
                            "Non-owned Trailer DED. Currency": trailer_ded_currency,
                            "Non-owned Trailer DED. Amount": trailer_ded_amount,
                            "Non-owned Trailer Amount Expiry Date (yyyy/mm/dd)": trailer_expiry,
                            "Additional insured": additional_insured,
                            "Certificate Holder": certificate_holder,
                            "Cancellation Notice Period (days)": cancellation_period
                        }
                        
                        # Add to dataframe
                        st.session_state.certificates = pd.concat([
                            st.session_state.certificates, 
                            pd.DataFrame([certificate_data])
                        ], ignore_index=True)
                        
                        st.success("Certificate saved successfully!")
        
        # Display all processed certificates
        if not st.session_state.certificates.empty:
            st.subheader("Processed Certificates")
            st.dataframe(st.session_state.certificates)
            
            # Add export functionality
            excel_data = export_to_excel(st.session_state.certificates)
            st.download_button(
                label="Download Certificates as Excel",
                data=excel_data,
                file_name="insurance_certificates.xlsx",
                mime="application/vnd.ms-excel"
            )

# Add custom CSS to style the app
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    /* Make the camera box smaller */
    .stCamera {
        width: 400px !important;
    }
    /* Adjust the camera container height */
    div[data-testid="stCamera"] > div {
        min-height: 300px !important;
    }
    /* Make video element smaller */
    div[data-testid="stCamera"] video {
        max-height: 300px !important;
    }
    /* Style form headers */
    h5 {
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid #eee;
    }
    /* Add padding to text inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        padding: 6px 12px;
    }
    /* Hide default streamlit subheader margins for custom sections */
    .main .block-container .element-container h3 {
        margin-top: 0;
    }
    /* Add more spacing between sections */
    .stForm {
        margin-top: 20px;
    }
    /* Status card styles */
    .status-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    .blue-card {
        background-color: #E8F0FE;
    }
    .red-card {
        background-color: #FEEAE6;
    }
    .card-icon {
        margin-right: 15px;
        font-size: 20px;
    }
    .card-text {
        display: flex;
        flex-direction: column;
    }
    .card-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()

