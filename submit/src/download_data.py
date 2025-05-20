#!/usr/bin/env python3
"""
download_data.py

Download and export text subsets for specified languages from FLORES-200 dataset.

Each line is:
  - UTF-8 NFC normalized
  - Left-to-right

Usage:
  pip install datasets
  python src/download_data.py
"""

import os
import unicodedata
import argparse
from typing import Dict, List
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize to NFC form
    text = unicodedata.normalize('NFC', text)
    
    # Remove excessive whitespace
    text = " ".join(text.split())

    return text

def get_flores_language_mapping() -> Dict[str, str]:
    """
    Get mapping between ISO language codes and FLORES dataset keys.
    FLORES uses different codes for some languages.
    """
    # This mapping helps translate between standard ISO codes and FLORES-specific codes
    return {
        'en':    'eng_Latn',
        'es':    'spa_Latn',
        'fr':    'fra_Latn',
        'de':    'deu_Latn',
        'ru':    'rus_Cyrl',
        'zh':    'zho_Hans',
        'ja':    'jpn_Jpan',
        'ar':    'arb_Arab',
        'ko':    'kor_Hang',
        'pt':    'por_Latn',
        'it':    'ita_Latn',
        'nl':    'nld_Latn',
        'pl':    'pol_Latn',
        'sv':    'swe_Latn',
        'da':    'dan_Latn',
        'fi':    'fin_Latn',
        'no':    'nob_Latn',
        'cs':    'ces_Latn',
        'hu':    'hun_Latn',
        'ro':    'ron_Latn',
        'el':    'ell_Grek',
        'bg':    'bul_Cyrl',
        'uk':    'ukr_Cyrl',
        'tr':    'tur_Latn',
        'he':    'heb_Hebr',
        'fa':    'pes_Arab',
        'vi':    'vie_Latn',
        'th':    'tha_Thai',
        'hi':    'hin_Deva',
        'bn':    'ben_Beng',
        'ta':    'tam_Taml',
        'ur':    'urd_Arab',
        'ms':    'msa_Latn',
        'id':    'ind_Latn',
        'sw':    'swh_Latn',
        'af':    'afr_Latn',
        'am':    'amh_Ethi',
        'hy':    'hye_Armn',
        'az':    'azj_Latn',
        'eu':    'eus_Latn',
        'be':    'bel_Cyrl',
        'ca':    'cat_Latn',
        'hr':    'hrv_Latn',
        'et':    'est_Latn',
        'gl':    'glg_Latn',
        'ka':    'kat_Geor',
        'gu':    'guj_Gujr',
        'ht':    'hat_Latn',
        'ha':    'hau_Latn',
        'is':    'isl_Latn',
        'ig':    'ibo_Latn',
        'ga':    'gle_Latn',
        'kk':    'kaz_Cyrl',
        'ky':    'kir_Cyrl',
        'lo':    'lao_Laoo',
        'lv':    'lav_Latn',
        'lt':    'lit_Latn',
        'lb':    'ltz_Latn',
        'mk':    'mkd_Cyrl',
        'mg':    'mlg_Latn',
        'ml':    'mal_Mlym',
        'mt':    'mlt_Latn',
        'mi':    'mri_Latn',
        'mr':    'mar_Deva',
        'mn':    'mon_Cyrl',
        'my':    'mya_Mymr',
        'ne':    'npi_Deva',
        'ps':    'pus_Arab',
        'si':    'sin_Sinh',
        'sk':    'slk_Latn',
        'sl':    'slv_Latn',
        'so':    'som_Latn',
        'tl':    'tgl_Latn',
        'tt':    'tat_Cyrl',
        'te':    'tel_Telu',
        'uz':    'uzb_Latn',
        'cy':    'cym_Latn',
        'xh':    'xho_Latn',
        'yi':    'yid_Hebr',
        'yo':    'yor_Latn',
        'zu':    'zul_Latn',
    }

def download_flores(iso_languages: List[str] = None):
    """
    Download and process FLORES-200 dataset.
    
    Args:
        iso_languages: List of ISO language codes to download. If None, download all languages.
    """
    logger.info("Downloading FLORES-200 dataset...")
    
    try:
        # Create flores subdirectory if it doesn't exist
        flores_dir = os.path.join(OUTPUT_DIR, "flores")
        os.makedirs(flores_dir, exist_ok=True)
        
        # Get language mapping
        lang_mapping = get_flores_language_mapping()
        
        # Filter languages if specified
        if iso_languages:
            # Only keep languages that are in the mapping
            filtered_mapping = {k: v for k, v in lang_mapping.items() if k in iso_languages}
            if not filtered_mapping:
                logger.error(f"None of the specified languages {iso_languages} found in FLORES mapping")
                return
            lang_mapping = filtered_mapping
        
        # Load FLORES dataset
        logger.info("Loading FLORES dataset...")
        flores = load_dataset("facebook/flores", "all")
        logger.info(f"FLORES dataset loaded. Available splits: {flores.keys()}")
        
        # Get a sample item to check the structure
        sample_item = next(iter(flores['dev']))
        logger.info(f"Sample item keys: {list(sample_item.keys())}")
        
        # Process each language in the mapping
        for iso_code, flores_code in lang_mapping.items():
            # Use the FLORES code for the filename
            output_file = os.path.join(flores_dir, f"{flores_code}.txt")
            logger.info(f"Processing FLORES-200 for {iso_code} ({flores_code})...")
            
            # Construct the key for this language in the dataset
            lang_key = f'sentence_{flores_code}'
            
            # Check if the key exists in the sample item
            if lang_key not in sample_item:
                logger.warning(f"Key '{lang_key}' not found in FLORES dataset. Available keys: {list(sample_item.keys())}")
                continue
            
            count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                # Process all available splits
                for split in flores.keys():
                    logger.info(f"Processing split: {split}")
                    for item in flores[split]:
                        try:
                            if lang_key in item:
                                text = normalize_text(item[lang_key])
                                if text:
                                    f.write(f"{text}\n")
                                    count += 1
                                    
                                    if count % 1000 == 0:
                                        logger.info(f"Processed {count} sentences for {flores_code}")
                        except Exception as e:
                            logger.warning(f"Error processing item for {flores_code}: {e}")
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
                        logger.info(f"Sample content for {flores_code}: {first_lines}")
            else:
                logger.error(f"File {output_file} was not created!")
    
    except Exception as e:
        logger.error(f"Error downloading FLORES-200: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Download and process FLORES-200 dataset")
    parser.add_argument("--languages", nargs="+", 
                        help="ISO language codes to download (e.g., en fr de). If not specified, all languages will be downloaded.")
    parser.add_argument("--list-languages", action="store_true", 
                        help="List all supported languages and exit")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode with more verbose output")
    
    args = parser.parse_args()
    
    if args.list_languages:
        print("Supported languages:")
        lang_mapping = get_flores_language_mapping()
        for iso_code, flores_code in sorted(lang_mapping.items()):
            print(f"  {iso_code}: {flores_code}")
        return
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Create output directory
    ensure_output_dir()
    
    # Download FLORES dataset
    download_flores(args.languages)
    
    logger.info("Data download and processing complete!")

if __name__ == "__main__":
    main()
