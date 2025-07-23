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


WIKIDATA_ENTITIES = {
    'ace': 'Q27683', 'acm': 'Q56232', 'acq': 'Q56579', 'aeb': 'Q56240', 'afr': 'Q14196',
    'als': 'Q387066', 'amh': 'Q28244', 'apc': 'Q56593', 'arb': 'Q13955', 'ars': 'Q56574',
    'ary': 'Q56426', 'arz': 'Q29919', 'asm': 'Q29401', 'ast': 'Q29507', 'awa': 'Q29579',
    'ayr': 'Q20526610', 'azb': 'Q3449805', 'azj': 'Q9292', 'bak': 'Q13389', 'bam': 'Q33243',
    'ban': 'Q33070', 'bel': 'Q9091', 'bem': 'Q33052', 'ben': 'Q9610', 'bho': 'Q33268',
    'bjn': 'Q33151', 'bod': 'Q34271', 'bos': 'Q9303', 'bug': 'Q33190', 'bul': 'Q7918',
    'cat': 'Q7026', 'ceb': 'Q33239', 'ces': 'Q9056', 'cjk': 'Q2422065', 'ckb': 'Q36811',
    'cmn': 'Q24841726', 'crh': 'Q33357', 'cym': 'Q9309', 'dan': 'Q9035', 'deu': 'Q188',
    'dik': 'Q56466', 'dyu': 'Q32706', 'dzo': 'Q33081', 'ekk': '', 'ell': 'Q9129',
    'eng': 'Q1860', 'epo': 'Q143', 'eus': 'Q8752', 'ewe': 'Q30005', 'fao': 'Q25258',
    'fij': 'Q33295', 'fil': 'Q33298', 'fin': 'Q1412', 'fon': 'Q33291', 'fra': 'Q150',
    'fur': 'Q33441', 'fuv': 'Q36129', 'gaz': 'Q12639015', 'gla': 'Q9314', 'gle': 'Q9142',
    'glg': 'Q9307', 'gug': '', 'guj': 'Q5137', 'hat': 'Q33491', 'hau': 'Q56475',
    'heb': 'Q9288', 'hin': 'Q1568', 'hne': 'Q33158', 'hrv': 'Q6654', 'hun': 'Q9067', 'hye': 'Q8785',
    'ibo': 'Q33578', 'ilo': 'Q35936', 'ind': 'Q9240', 'isl': 'Q294', 'ita': 'Q652',
    'jav': 'Q33549', 'jpn': 'Q5287', 'kab': 'Q35853', 'kac': 'Q33332', 'kam': 'Q33587',
    'kan': 'Q33673', 'kas': 'Q33552', 'kat': 'Q8108', 'kaz': 'Q9252', 'kbp': 'Q35475',
    'kea': 'Q35963', 'khk': '', 'khm': 'Q9205', 'kik': 'Q33587', 'kin': 'Q33573', 'kir': 'Q9255',
    'kmb': 'Q35891', 'kmr': 'Q36163', 'knc': 'Q15637215', 'kor': 'Q9176', 'ktu': '',
    'lao': 'Q9211', 'lij': 'Q36106', 'lim': 'Q102172', 'lin': 'Q36217', 'lit': 'Q9083',
    'lmo': 'Q33754', 'ltg': 'Q36212', 'ltz': 'Q9051', 'lua': 'Q34173', 'lug': 'Q33368',
    'luo': 'Q5414796', 'lus': 'Q36147', 'lvs': 'Q9078', 'mag': 'Q33728', 'mai': 'Q36109',
    'mal': 'Q9237', 'mar': 'Q1571', 'min': 'Q13324', 'mkd': 'Q9296', 'mlt': 'Q9166',
    'mni': 'Q33868', 'mos': 'Q36096', 'mri': 'Q36451', 'mya': 'Q9228', 'nld': 'Q7411',
    'nno': 'Q9043', 'nob': 'Q9043', 'npi': 'Q33823', 'nso': 'Q33890', 'nus': 'Q33675',
    'nya': 'Q33273', 'oci': 'Q14185', 'ory': 'Q33810', 'pag': 'Q33879', 'pan': 'Q58635',
    'pap': 'Q33856', 'pbt': 'Q58680', 'pes': 'Q3513637', 'plt': 'Q15069308', 'pol': 'Q809',
    'por': 'Q5146', 'prs': 'Q178440', 'quy': 'Q3573199', 'ron': 'Q7913', 'run': 'Q33583',
    'rus': 'Q7737', 'sag': 'Q33954', 'san': 'Q11059', 'sat': 'Q33965', 'scn': 'Q33973',
    'shn': 'Q56482', 'sin': 'Q13267', 'slk': 'Q9058', 'slv': 'Q9063', 'smo': 'Q34011',
    'sna': 'Q34004', 'snd': 'Q33997', 'som': 'Q13275', 'sot': 'Q34340', 'spa': 'Q1321',
    'srd': 'Q33976', 'srp': 'Q9299', 'ssw': 'Q34014', 'sun': 'Q34002', 'swe': 'Q9027',
    'swh': 'Q7838', 'szl': 'Q30319', 'tam': 'Q5885', 'taq': 'Q4670066', 'tat': 'Q25285',
    'tel': 'Q8097', 'tgk': 'Q9260', 'tha': 'Q9217', 'tir': 'Q34124', 'tpi': 'Q34159',
    'tsn': 'Q34137', 'tso': 'Q34327', 'tuk': 'Q9267', 'tum': 'Q34138', 'tur': 'Q256',
    'twi': 'Q36850', 'uig': 'Q13263', 'ukr': 'Q8798', 'umb': 'Q36983', 'urd': 'Q1617',
    'uzn': 'Q9264', 'vec': 'Q32724', 'vie': 'Q9199', 'war': 'Q34279', 'wol': 'Q34257',
    'xho': 'Q13218', 'ydd': 'Q8641', 'yor': 'Q34311', 'yue': 'Q7033959', 'zgh': 'Q7850',
    'zsm': 'Q9237', 'zul': 'Q10179', "alb": "Q8748", "ber": "Q25448", "cnt": "Q9186",
    "grn": "Q35876", "idu": "Q35224", "lat": "Q397", "ndb": "Q35613", "nqo": "Q2494019",
    "icl": "Q294",
}

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
    'co': None,  # 'cos_Latn',
    'cs': 'ces_Latn',
    'cy': 'cym_Latn',
    'da': 'dan_Latn',
    'de': 'deu_Latn',
    'el': 'ell_Grek',
    'el-Latn': 'ell_Grek',
    'en': 'eng_Latn',
    'eo': 'epo_Latn',
    'es': 'spa_Latn',
    'et': None,  # 'est_Latn',
    'eu': 'eus_Latn',
    'fa': 'pes_Arab',
    'fi': 'fin_Latn',
    'fil': 'fil_Latn',
    'fr': 'fra_Latn',
    'fy': None,  # 'fry_Latn',
    'ga': 'gle_Latn',
    'gd': 'gla_Latn',
    'gl': 'glg_Latn',
    'gu': 'guj_Gujr',
    'ha': 'hau_Latn',
    'haw': None,  # 'haw_Latn',
    'hi': 'hin_Deva',
    'hi-Latn': 'hin_Deva',
    'hmn': None,  # 'hmn_Latn',
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
    'la': None,  # 'lat_Latn',
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


lang_iso_replacements = {
    "arb": "ara",
    "icl": "isl",
    "zsm": "msa",
}
