import random
from collections import defaultdict
from pathlib import Path
import pickle
import os
import logging
import math
import glob

import datasets
import torch
from transformers import CanineTokenizer
from sklearn.model_selection import train_test_split
import tqdm

PROJECT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "trainer_output"


def get_tokenized_inputs_path(max_length):
    return PROJECT_PATH / "trainer_output" / f"tokenized_inputs_{max_length}.pkl"


def create_language_dict(texts: list[str], labels: list[str]):
    languages: defaultdict[str, list[str]] = defaultdict(lambda: [])
    for idx, label in enumerate(labels):
        if isinstance(texts[idx], str):
            languages[label].append(texts[idx])

    return languages


def get_data():
    text_path = DATA_PATH / "text.pkl"
    label_path = DATA_PATH / "label.pkl"

    if not os.path.exists(text_path) or not os.path.exists(label_path):
        dataset = datasets.load_dataset(
            'laurievb/OpenLID-v2', token=os.environ.get("HUGGINGFACE_TOKEN"),
            features=datasets.Features({  # Present because without it, the function throws an exception
                'text': datasets.Value('string'),
                'language': datasets.Value('string'),
                'source': datasets.Value('string'),
                '__index_level_0__': datasets.Value('int64')
            })
        )
        df = dataset["train"]
        del dataset

        # df = df.select(range(1_000_000))
        df = df.filter(lambda d: isinstance(d['text'], str))

        logging.info("Splitting labels and texts...")
        texts, labels = df['text'], df['language']
        save_object(texts, text_path)
        save_object(labels, label_path)

        return texts, labels

    else:
        logging.info("Loading data...")
        return load_object(text_path), load_object(label_path)


def load_dataset(
    samples_count: int | None,
    test_size: float = 0.05
):
    """Load OpenLID dataset"""
    texts, labels = get_data()
    if samples_count:
        texts, labels = sample_dataset(
            create_language_dict(texts, labels), samples_count)

    logging.info("Splitting dataset...")
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
    )

    return train_texts, eval_texts, train_labels, eval_labels


def sample_dataset(languages: dict[str, list[str]], samples_per_language: int):
    logging.info("Sampling %s samples per language", samples_per_language)
    new_texts = []
    new_labels = []
    for language, lang_texts in languages.items():
        if len(lang_texts) <= samples_per_language:
            new_texts.extend(lang_texts)
            new_labels.extend([language] * len(lang_texts))
        else:
            new_texts.extend(random.sample(lang_texts, k=samples_per_language))
            new_labels.extend([language] * samples_per_language)

    return new_texts, new_labels


def tokenize_input(texts: list[str], tokenizer: CanineTokenizer, max_length=512):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')


def tokenize_dataset(texts, tokenizer: CanineTokenizer, max_length=512):
    if os.path.exists(get_tokenized_inputs_path(max_length)):
        tokenized = load_object(get_tokenized_inputs_path(max_length))
    else:
        tokenized = [tokenize_input([text], tokenizer, max_length)
                     for text in tqdm.tqdm(texts) if isinstance(text, str)]
        save_object(tokenized, get_tokenized_inputs_path(max_length))

    return tokenized


def save_object(obj, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_eval_steps(dataset: torch.utils.data.Dataset, batch_size, epochs, evals):
    steps = math.ceil(len(dataset) / batch_size) * epochs
    return math.floor(steps / evals)


def flores_to_iso(flores_label: str):
    return str(flores_label[:3])


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the given directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        return None

    # Extract step numbers and find the latest one
    def get_step_number(checkpoint_path):
        try:
            return int(os.path.basename(checkpoint_path).split('-')[1])
        except (IndexError, ValueError):
            return 0

    latest_checkpoint = max(checkpoints, key=get_step_number)
    return latest_checkpoint


def get_checkpoint(no_resume: bool, checkpoint_path: str | None, model_path: str):
    """
    Get the checkpoint from which we start training

    Args:
        no_resume: Start from scratch
        checpoint_path: Specific checkpoint to start from
        model_path: The final path to save the model to

    Returns:
        The path to the checkpoint to start training from or None if we are to start from scratch
    """
    if no_resume:
        return None

    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        else:
            logging.warning(
                "Specified checkpoint path does not exist: %s", checkpoint_path)
            return None

    return find_latest_checkpoint(model_path)


class KeyDict(dict):
    def __init__(self, *args, **kwargs):
        classes = kwargs.pop('classes', [])
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        self.classes = classes

    def __missing__(self, key):
        stripped_key = key[9:] if key.startswith("__label__") else key
        if stripped_key in self.classes:
            return stripped_key

        return None


OPENLID_CLASSES = [
    'ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn',
    'als_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab',
    'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab',
    'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn',
    'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn',
    'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn',
    'ckb_Arab', 'cmn_Hans', 'cmn_Hant', 'crh_Latn', 'cym_Latn', 'dan_Latn',
    'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ekk_Latn', 'ell_Grek',
    'eng_Latn', 'epo_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'fij_Latn',
    'fil_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn',
    'gaz_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'gug_Latn', 'guj_Gujr',
    'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn',
    'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn',
    'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn',
    'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'kaz_Cyrl', 'kbp_Latn',
    'kea_Latn', 'khk_Cyrl', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl',
    'kmb_Latn', 'kmr_Latn', 'knc_Arab', 'knc_Latn', 'kor_Hang', 'ktu_Latn',
    'lao_Laoo', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn',
    'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn',
    'lvs_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn',
    'mkd_Cyrl', 'mlt_Latn', 'mni_Beng', 'mos_Latn', 'mri_Latn', 'mya_Mymr',
    'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn',
    'nya_Latn', 'oci_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn',
    'pbt_Arab', 'pes_Arab', 'plt_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab',
    'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva',
    'sat_Olck', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn',
    'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn',
    'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn',
    'szl_Latn', 'tam_Taml', 'taq_Latn', 'taq_Tfng', 'tat_Cyrl', 'tel_Telu',
    'tgk_Cyrl', 'tha_Thai', 'tir_Ethi', 'tpi_Latn', 'tsn_Latn', 'tso_Latn',
    'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'uig_Arab', 'ukr_Cyrl',
    'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn',
    'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zgh_Tfng',
    'zsm_Latn', 'zul_Latn'
]


FASTTEXT_TO_OPENLID = KeyDict(classes=OPENLID_CLASSES)
GLOT_TO_OPENLID = KeyDict(classes=OPENLID_CLASSES)
OPENLID_TO_OPENLID = KeyDict(classes=OPENLID_CLASSES)
GCLD_TO_OPENLID = {
    'af': 'afr_Latn',
    'am': 'amh_Ethi',
    'ar': 'arb_Arab',
    'az': 'azb_Arab',
    'be': 'bel_Cyrl',
    'bg': 'bul_Cyrl',
    'bg-Latn': 'bul_Cyrl',
    'bn': 'ben_Beng',
    'bs': 'bos_Latn',
    'ca': 'cat_Latn',
    'ceb': 'ceb_Latn',
    'co': None, #'cos_Latn',
    'cs': 'ces_Latn',
    'cy': 'cym_Latn',
    'da': 'dan_Latn',
    'de': 'deu_Latn',
    'el': 'ell_Grek',
    'el-Latn': 'ell_Grek',
    'en': 'eng_Latn',
    'eo': 'epo_Latn',
    'es': 'spa_Latn',
    'et': None, # 'est_Latn',
    'eu': 'eus_Latn',
    'fa': 'pes_Arab',
    'fi': 'fin_Latn',
    'fil': 'fil_Latn',
    'fr': 'fra_Latn',
    'fy': None, # 'fry_Latn',
    'ga': 'gle_Latn',
    'gd': 'gla_Latn',
    'gl': 'glg_Latn',
    'gu': 'guj_Gujr',
    'ha': 'hau_Latn',
    'haw': None, # 'haw_Latn',
    'hi': 'hin_Deva',
    'hi-Latn': 'hin_Deva',
    'hmn': None, # 'hmn_Latn',
    'hr': 'hrv_Latn',
    'ht': 'hat_Latn',
    'hu': 'hun_Latn',
    'hy': 'hye_Armn',
    'id': 'ind_Latn',
    'ig': 'ibo_Latn',
    'is': 'isl_Latn',
    'it': 'ita_Latn',
    'iw': 'heb_Hebr',
    'ja': 'jpn_Jpan',
    'ja-Latn': 'jpn_Jpan',
    'jv': 'jav_Latn',
    'ka': 'kat_Geor',
    'kk': 'kaz_Cyrl',
    'km': 'khm_Khmr',
    'kn': 'kan_Knda',
    'ko': 'kor_Hang',
    'ku': 'kmr_Latn',
    'ky': 'kir_Cyrl',
    'la': None, # 'lat_Latn',
    'lb': 'ltz_Latn',
    'lo': 'lao_Laoo',
    'lt': 'lit_Latn',
    'lv': 'lij_Latn',
    'mg': 'plt_Latn',
    'mi': 'mri_Latn',
    'mk': 'mkd_Cyrl',
    'ml': 'mal_Mlym',
    'mn': 'khk_Cyrl',
    'mr': 'mar_Deva',
    'ms': 'zsm_Latn',
    'mt': 'mlt_Latn',
    'my': 'mya_Mymr',
    'ne': 'npi_Deva',
    'nl': 'nld_Latn',
    'no': 'nob_Latn',
    'ny': 'nya_Latn',
    'pa': 'pan_Guru',
    'pl': 'pol_Latn',
    'ps': 'pbt_Arab',
    'pt': 'por_Latn',
    'ro': 'ron_Latn',
    'ru': 'rus_Cyrl',
    'ru-Latn': 'rus_Cyrl',
    'sd': 'snd_Arab',
    'si': 'sin_Sinh',
    'sk': 'slk_Latn',
    'sl': 'slv_Latn',
    'sm': 'smo_Latn',
    'sn': 'sna_Latn',
    'so': 'som_Latn',
    'sq': 'als_Latn',
    'sr': 'srp_Cyrl',
    'st': 'sot_Latn',
    'su': 'sun_Latn',
    'sv': 'swe_Latn',
    'sw': 'swh_Latn',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'tg': 'tgk_Cyrl',
    'th': 'tha_Thai',
    'tr': 'tur_Latn',
    'uk': 'ukr_Cyrl',
    'ur': 'urd_Arab',
    'uz': 'uzn_Latn',
    'vi': 'vie_Latn',
    'xh': 'xho_Latn',
    'yi': 'ydd_Hebr',
    'yo': 'yor_Latn',
    'zh': 'cmn_Hans',
    'zh-Latn': 'cmn_Hans',
    'zu': 'zul_Latn'
}