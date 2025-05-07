#!/usr/bin/env python3
"""
download_data.py

Download and export text subsets for specified languages from FLORES-200 dataset.

Each line is:
  - UTF-8 NFC normalized
  - ≤ 300 characters
  - Left-to-right

Usage:
  pip install datasets
  python src/download_data.py
"""

import os
import unicodedata
import argparse
from typing import List, Dict
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    # Major languages
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ar': 'Arabic',
    
    # European languages
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'sv': 'Swedish',
    'da': 'Danish',
    'fi': 'Finnish',
    'no': 'Norwegian',
    'cs': 'Czech',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'el': 'Greek',
    'bg': 'Bulgarian',
    'uk': 'Ukrainian',
    
    # Asian languages
    'ko': 'Korean',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'ur': 'Urdu',
    'fa': 'Persian',
    'id': 'Indonesian',
    'ms': 'Malay',
    
    # Other languages
    'tr': 'Turkish',
    'he': 'Hebrew',
    'sw': 'Swahili',
    'af': 'Afrikaans',
    'am': 'Amharic',
    'hy': 'Armenian',
    'az': 'Azerbaijani',
    'eu': 'Basque',
    'be': 'Belarusian',
    'ca': 'Catalan',
    'hr': 'Croatian',
    'et': 'Estonian',
    'gl': 'Galician',
    'ka': 'Georgian',
    'gu': 'Gujarati',
    'ht': 'Haitian Creole',
    'ha': 'Hausa',
    'is': 'Icelandic',
    'ig': 'Igbo',
    'ga': 'Irish',
    'kk': 'Kazakh',
    'ky': 'Kyrgyz',
    'lo': 'Lao',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'lb': 'Luxembourgish',
    'mk': 'Macedonian',
    'mg': 'Malagasy',
    'ml': 'Malayalam',
    'mt': 'Maltese',
    'mi': 'Maori',
    'mr': 'Marathi',
    'mn': 'Mongolian',
    'my': 'Myanmar (Burmese)',
    'ne': 'Nepali',
    'ps': 'Pashto',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'so': 'Somali',
    'tl': 'Tagalog',
    'tt': 'Tatar',
    'te': 'Telugu',
    'uz': 'Uzbek',
    'cy': 'Welsh',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'zu': 'Zulu',
}

# Output directory
OUTPUT_DIR = "data"

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Using data directory: {OUTPUT_DIR}")

def normalize_text(text: str) -> str:
    """
    Normalize text:
    - Convert to NFC form
    - Remove excessive whitespace
    - Trim to 300 characters
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize to NFC form
    text = unicodedata.normalize('NFC', text)
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Trim to 300 characters
    if len(text) > 300:
        text = text[:300]
    
    return text

def get_flores_language_mapping() -> Dict[str, str]:
    """
    Get mapping between ISO language codes and FLORES dataset keys.
    FLORES uses different codes for some languages.
    """
    # This mapping helps translate between standard ISO codes and FLORES-specific codes
    return {
        # Standard ISO -> FLORES code
        'zh': 'zho_Hans',  # Simplified Chinese
        'zh-tw': 'zho_Hant',  # Traditional Chinese
        'en': 'eng',
        'fr': 'fra',
        'de': 'deu',
        'es': 'spa',
        'ru': 'rus',
        'ja': 'jpn',
        'ar': 'arb',  # Modern Standard Arabic
        'ko': 'kor',
        'pt': 'por',
        'it': 'ita',
        'nl': 'nld',
        'pl': 'pol',
        'tr': 'tur',
        'fa': 'pes',  # Persian
        'vi': 'vie',
        'hi': 'hin',
        'th': 'tha',
        'uk': 'ukr',
        'sv': 'swe',
        'cs': 'ces',
        'ro': 'ron',
        'hu': 'hun',
        'el': 'ell',
        'da': 'dan',
        'fi': 'fin',
        'bg': 'bul',
        'hr': 'hrv',
        'he': 'heb',
        'sk': 'slk',
        'no': 'nob',  # Norwegian Bokmål
        'id': 'ind',
        'ca': 'cat',
        'lt': 'lit',
        'sl': 'slv',
        'lv': 'lav',
        'et': 'est',
        'ta': 'tam',
        'ur': 'urd',
        'bn': 'ben',
        'ms': 'msa',
        'am': 'amh',
        'sw': 'swh',
        'te': 'tel',
        'mr': 'mar',
        'ml': 'mal',
        'gu': 'guj',
        'af': 'afr',
        'my': 'mya',
        'ne': 'npi',
        'si': 'sin',
        'km': 'khm',
        'eu': 'eus',
        'gl': 'glg',
        'hy': 'hye',
        'az': 'azj',
        'be': 'bel',
        'is': 'isl',
        'mk': 'mkd',
        'ka': 'kat',
        'mn': 'khk',
        'cy': 'cym',
        'kk': 'kaz',
        'uz': 'uzb',
        'ps': 'pus',
        'ha': 'hau',
        'yo': 'yor',
        'zu': 'zul',
        'xh': 'xho',
        'ig': 'ibo',
        'so': 'som',
        'mg': 'mlg',
        'mt': 'mlt',
        'mi': 'mri',
        'lo': 'lao',
        'ky': 'kir',
        'tt': 'tat',
        'ga': 'gle',
        'lb': 'ltz',
        'ht': 'hat',
        'tl': 'tgl',
        'yi': 'yid',
    }

def download_flores(languages: List[str]):
    """Download and process FLORES-200 dataset."""
    logger.info("Downloading FLORES-200 dataset...")
    
    try:
        # Create flores subdirectory if it doesn't exist
        flores_dir = os.path.join(OUTPUT_DIR, "flores")
        os.makedirs(flores_dir, exist_ok=True)
        
        # Get language mapping
        lang_mapping = get_flores_language_mapping()
        
        # Load FLORES dataset
        logger.info("Loading FLORES dataset...")
        
        # Try different dataset IDs
        try:
            # Try the new version first
            flores = load_dataset("facebook/flores", "all")
            logger.info("Successfully loaded FLORES dataset with 'facebook/flores'")
        except Exception as e:
            logger.warning(f"Failed to load with 'facebook/flores': {e}")
            try:
                # Try the legacy version
                flores = load_dataset("gsarti/flores_101")
                logger.info("Successfully loaded FLORES dataset with 'gsarti/flores_101'")
            except Exception as e2:
                logger.warning(f"Failed to load with 'gsarti/flores_101': {e2}")
                # Try one more alternative
                flores = load_dataset("facebook/flores", "flores200")
                logger.info("Successfully loaded FLORES dataset with 'facebook/flores', 'flores200'")
        
        logger.info(f"FLORES dataset loaded. Available splits: {flores.keys()}")
        
        # Determine the correct key format based on the dataset version
        sample_item = next(iter(flores['dev']))
        logger.info(f"Sample item keys: {list(sample_item.keys())}")
        
        # Process each language
        for lang_code in languages:
            output_file = os.path.join(flores_dir, f"{lang_code}.txt")
            logger.info(f"Processing FLORES-200 for {SUPPORTED_LANGUAGES.get(lang_code, lang_code)}...")
            
            # Try different key formats for this language
            possible_keys = []
            
            # Try with the FLORES-specific code if available
            if lang_code in lang_mapping:
                flores_code = lang_mapping[lang_code]
                possible_keys.extend([
                    f'sentence_{flores_code}',
                    flores_code,
                    f'{flores_code}_sentence',
                    f'flores.{flores_code}'
                ])
            
            # Also try with the original code
            possible_keys.extend([
                f'sentence_{lang_code}',
                lang_code,
                f'{lang_code}_sentence',
                f'flores.{lang_code}'
            ])
            
            # Find the correct key for this language
            lang_key = None
            for key in possible_keys:
                if key in sample_item:
                    lang_key = key
                    logger.info(f"Found language key: {lang_key}")
                    break
            
            if not lang_key:
                # Try to find a key that contains the language code
                for key in sample_item.keys():
                    if lang_code in key or (lang_code in lang_mapping and lang_mapping[lang_code] in key):
                        lang_key = key
                        logger.info(f"Found language key by partial match: {lang_key}")
                        break
            
            if not lang_key:
                logger.warning(f"Could not find key for language '{lang_code}' in FLORES dataset. Available keys: {list(sample_item.keys())}")
                continue
            
            count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                # Process all available splits
                for split in flores.keys():
                    logger.info(f"Processing split: {split}")
                    for item in flores[split]:
                        try:
                            # Try to get the text using the identified key
                            if lang_key in item:
                                text = item[lang_key]
                            elif '.' in lang_key:
                                # Handle nested keys
                                parts = lang_key.split('.')
                                text = item
                                for part in parts:
                                    text = text[part]
                            else:
                                continue
                                
                            text = normalize_text(text)
                            if text:
                                f.write(f"{text}\n")
                                count += 1
                                
                                if count % 1000 == 0:
                                    logger.info(f"Processed {count} sentences for {lang_code}")
                        except Exception as e:
                            logger.warning(f"Error processing item for {lang_code}: {e}")
                            continue
            
            logger.info(f"Saved {count} FLORES-200 sentences to {output_file}")
            
            # Verify the file was created and has content
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"File {output_file} created with size: {file_size} bytes")
                
                # Read a few lines to verify content
                if count > 0:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        first_lines = [f.readline().strip() for _ in range(min(5, count))]
                        logger.info(f"Sample content for {lang_code}: {first_lines}")
            else:
                logger.error(f"File {output_file} was not created!")
    
    except Exception as e:
        logger.error(f"Error downloading FLORES-200: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Download and process FLORES-200 dataset")
    parser.add_argument("--languages", nargs="+", default=list(SUPPORTED_LANGUAGES.keys()),
                        help=f"Language codes to download. Available: {', '.join(SUPPORTED_LANGUAGES.keys())}")
    parser.add_argument("--list-languages", action="store_true", 
                        help="List all supported languages and exit")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode with more verbose output")
    
    args = parser.parse_args()
    
    if args.list_languages:
        print("Supported languages:")
        for code, name in sorted(SUPPORTED_LANGUAGES.items()):
            print(f"  {code}: {name}")
        return
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Validate languages
    invalid_langs = set(args.languages) - set(SUPPORTED_LANGUAGES.keys())
    if invalid_langs:
        logger.warning(f"Unsupported languages: {', '.join(invalid_langs)}")
        args.languages = list(set(args.languages) & set(SUPPORTED_LANGUAGES.keys()))
    
    if not args.languages:
        logger.error("No valid languages specified")
        return
    
    logger.info(f"Processing languages: {', '.join(args.languages)}")
    
    # Create output directory
    ensure_output_dir()
    
    # Download FLORES dataset
    download_flores(args.languages)
    
    # Verify data was downloaded
    total_files = 0
    total_lines = 0
    flores_dir = os.path.join(OUTPUT_DIR, "flores")
    if os.path.exists(flores_dir):
        for file in os.listdir(flores_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(flores_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        total_lines += line_count
                        total_files += 1
                        logger.info(f"File {file_path}: {line_count} lines")
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
    
    logger.info(f"Data download and processing complete! Downloaded {total_files} files with {total_lines} total lines.")

if __name__ == "__main__":
    main()
