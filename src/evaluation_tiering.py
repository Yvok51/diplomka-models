from collections import defaultdict
import argparse
import sys
import typing
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import random
import os

import tqdm

from common import WIKIDATA_ENTITIES, flores_to_iso, save_object, load_object, PROJECT_PATH

SPEAKERS_PATH = PROJECT_PATH / "trainer_output" / "speakers.pkl"


LANGUAGE_TIERS = [
    # 0
    "ber,nan,sin,ndb,vmw,kau,ary,fry,acm,acq,aeb,ajp,apc,ars,ace,als,awa,bem,ban,cjk,dik,dyu,fon,fuv,gaz,hne,kac,kam,kbp,kea,khk,kmb,knc,lim,lua,luo,lus,mag,mos,nqo,nso,nus,pbt,plt,quy,shn,sot,taq,umb,war",

    # 1
    # Albanian, Assamese, Azerbaijani, Bambara, Burmese, Esperanto, Igbo, Javanese, Khmer, Kikuyu, Lingala, Luxembourgish, Maori, Norwegian, Occitan, Quechua, Samoan, Sango, Sardinian, Scottish, Sindhi, Somali, Swati, Telugu, Tibetan, Tok, Tsonga, Twi, Waray, Welsh
    "kin,kir,quz,mkd,ibo,mri,mya,nor,tel,jav,sun,azj,sdc,cnt,bre,mon,orm,min,hye,che,mlg,nep,alb,epo,oci,idu,cym,mal,kan,tgk,nno,nob,bak,tat,vol,hat,ina,chv,nav,arg,aka,asm,ast,ayr,azb,bam,bho,bjn,bod,bug,crh,dzo,fao,fij,fur,gla,grn,guj,ilo,kab,kas,khm,kmr,ckb,kik,kon,lij,lin,lmo,ltg,lug,mai,mni,nya,ory,pag,pap,run,ltz,sag,sat,scn,smo,sna,snd,som,srd,ssw,szl,tpi,tso,tuk,tum,twi,uig,vec,ydd,ewe",

    # 2
    # Amharic, Haitian, Hausa, Icelandic, Irish, Lao, Maltese, Marathi, Punjabi, Sanskrit, Swahili, Tigrinya, Tswana, Wolof, Xhosa, Yoruba, Zulu
    "amh,hau,swh,tsn,xho,yor,zul,mar,gle,pan,icl,lao,mlt,san,tir,wol",

    # 3
    # Afrikaans, Bangla, Belarusian, Bosnian, Bulgarian, Cebuano, Danish, Egyptian, Estonian, Galician, Georgian, Greek, Indonesian, Kazakh, Latin, Latvian, Lithuanian, Malay, Romanian, Slovak, Slovenian, Tagalog, Tamil, Thai, Ukrainian, Urdu, Uzbek, Hebrew
    "afr,arz,ben,bul,dan,ekk,ell,fil,tgl,heb,ind,kaz,ron,tam,tha,ukr,urd,uzb,zsm,kat,lat,slv,slk,lav,lit,bel,glg,bos,ceb",

    # 4
    # Basque, Catalan, Croatian, Czech, Dutch, Finnish, Hindi, Hungarian, Italian, Korean, Persian, Polish, Portuguese, Russian, Serbian, Swedish, Turkish, Vietnamese
    "eus,cat,hrv,ces,nld,fin,hin,hun,ita,kor,pes,pol,por,rus,srp,swe,tur,vie",

    # 5
    # Arabic, Chinese, English, French, German, Japanese, Spanish
    "arb,cmn,eng,fra,deu,jpn,spa",

    # from here
    # https://microsoft.github.io/linguisticdiversity/assets/lang2tax.txt

]
LANGUAGE_TIERS = [lang.split(",") for lang in LANGUAGE_TIERS]

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

first_language_query = """
    ?language p:P1098 [
      ps:P1098 ?speakers;
      pq:P518 wd:Q36870;
      pq:P585 ?point
    ].
"""


def collect_and_flatten_sparql_results(results, lang_names, tier_dict):
    lang_map = {}
    for binding in results['results']['bindings']:
        iso_code = binding["iso_code"]["value"]
        n_speakers = binding["speakers"]["value"]
        point = binding["point"]["value"]

        reported = lang_map.get(iso_code, [])
        reported.append((n_speakers, point))
        lang_map[iso_code] = reported

    lang_props = {}
    for lang, speakers_list in lang_map.items():
        speakers_list.sort(key=lambda x: x[1], reverse=True)
        n_speakers = speakers_list[0][0]

        lang_props[lang] = {
            "name": lang_names.get(lang, "Unknown"),
            "tier": tier_dict.get(lang, 0),
            "speakers": n_speakers,
        }
    return lang_props


def collect_number_of_speakers(lang_codes, lang_names, tier_dict):
    listed_langs = '{("' + '") ("'.join(lang_codes) + '")}'
    values_listed = "VALUES (?iso_code) " + listed_langs

    query = "SELECT * { " + values_listed + \
        "\n ?language wdt:P220 ?iso_code. " + first_language_query + " }"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    lang_props = collect_and_flatten_sparql_results(
        results, lang_names, tier_dict)
    return lang_props


def get_number_of_speakers_from_entity(entity_id):
    """Get the number of speakers for a given language"""
    query = "SELECT ?speakers WHERE { VALUES ?s {wd:entity_id} ?s wdt:P1098 ?speakers }"
    run_query = query.replace('entity_id', entity_id)

    sparql.setQuery(run_query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    if len(results['results']['bindings']) > 0:
        result = results['results']['bindings'][0]
        n_speakers = result["speakers"]["value"]
    else:
        n_speakers = 0
    return n_speakers


def collect_speakers() -> dict[str, int]:
    """Gather number of speakers for languages"""
    if os.path.exists(SPEAKERS_PATH):
        return load_object(SPEAKERS_PATH)

    speakers = {}
    for lang, entity_id in tqdm.tqdm(WIKIDATA_ENTITIES.items(), "Gathering speaker counts"):
        if entity_id:
            speakers[lang] = int(get_number_of_speakers_from_entity(entity_id))
            time.sleep(random.uniform(0.5, 1))
        else:
            speakers[lang] = 0

    save_object(speakers, SPEAKERS_PATH)

    return speakers

def language_weights_by_speaker_count() -> dict[str, float]:
    speakers = collect_speakers()
    total_speakers = sum(speakers.values())
    return {k: v / total_speakers for k, v in speakers.items()}


def read_flores_results(file: typing.TextIO) -> dict[str, dict[str, float]]:
    """Read a FLORES metric result file and return it grouped by the metric"""
    metrics = defaultdict(dict)
    for line in file:
        [metric, label, score] = line.split(",")
        metrics[metric][label] = float(score)
    return metrics

def get_label_tier(label: str, tiers: list[set[str]]):
    """Get the tier of the language represented by the label."""
    label = flores_to_iso(label) # remove script tag and leave only iso code
    for tier, languages in enumerate(tiers):
        if label in languages:
            return tier

    return 1 # tier 1 is default according to the paper

def group_by_tier(metrics: dict[str, dict[str, float]], tiers: list[set[str]]) -> dict[str, list[list[float]]]:
    """Group the metric results by tiers."""
    tier_scores = defaultdict(lambda: [[] for _ in range(len(tiers))])
    for metric, scores in metrics.items():
        for label, score in scores.items():
            tier_scores[metric][get_label_tier(label, tiers)].append(score)

    return tier_scores

def weighted_results(metrics: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    """Weight the metrics by the given weights and add such weighted averages"""
    weighted = {}
    for metric, scores in metrics.items():
        weighted[metric] = 0
        for label, score in scores.items():
            weighted[metric] += score * weights.get(flores_to_iso(label), 0)

    return weighted

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of language prediction using finetuned CANINE model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tiered_parser = subparsers.add_parser("tiered", help="Tier the results according to the language prevalence")
    tiered_parser.add_argument("--results", type=argparse.FileType('r'), nargs='*', default=[sys.stdin],
                        help="The result files to draw from")
    tiered_parser.add_argument("--output", type=argparse.FileType('w'), nargs='*',
                        default=[sys.stdout], help="The files to write the metrics to")

    weighted_parser = subparsers.add_parser("weighted", help="Weigh the results according to the number of speakers of the languages")
    weighted_parser.add_argument("--results", type=argparse.FileType('r'), nargs='*', default=[sys.stdin],
                        help="The result files to draw from")
    weighted_parser.add_argument("--output", type=argparse.FileType('w'), nargs='*',
                        default=[sys.stdout], help="The files to write the metrics to")

    args = parser.parse_args()

    assert len(args.results) == len(args.output) or len(args.output) == 1

    language_tiers = [set(tier) for tier in LANGUAGE_TIERS]

    for idx, file in enumerate(args.results):
        metrics = read_flores_results(file)

        output_file = args.output[idx] if len(args.output) > 1 else args.output[0]

        if args.command == "tiered":
            results = group_by_tier(metrics, language_tiers)
            for metric, tiers in results.items():
                for tier, scores in enumerate(tiers):
                    print(f"{metric},{tier},{sum(scores) / len(scores)}", file=output_file)

        if args.command == "weighted":
            results = weighted_results(metrics, language_weights_by_speaker_count())
            for metric, score in results.items():
                print(f"{metric},{score}", file=output_file)

if __name__ == "__main__":
    main()