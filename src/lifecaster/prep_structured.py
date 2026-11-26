# %%
import os

NUM_CORES = "8"
os.environ["POLARS_MAX_THREADS"] = NUM_CORES  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = NUM_CORES  # MUST BE BEFORE POLARS IMPORT
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["ARROW_NUM_THREADS"] = NUM_CORES

import pyarrow as pa

pa.set_cpu_count(int(NUM_CORES))

import json  # noqa: E402
import time  # noqa: E402

import polars as pl  # noqa: E402
from lifecaster.loggers import configure_logger  # noqa: E402
from lifecaster.paths import FPATH, check_and_copy_file_or_dir  # noqa: E402
from lifecaster.preprocessing import extract_specs, process_dataset  # noqa: E402
from loguru import logger  # noqa: E402

# %% [markdown]
# # WHERE TO SAVE OUR FILES

DATA_FOLDER = FPATH.DATA / "lifecaster"
DATA_FOLDER.mkdir(exist_ok=True)
SAMPLE_FOLDER = "lifecaster"
sample_path = FPATH.NETWORK_DATA / SAMPLE_FOLDER
check_and_copy_file_or_dir(sample_path)
# %%

CHUNK_SIZE = 50_000_000

# %%

log_path = FPATH.LOGS / "data_logs" / f"sequential_prep_{DATA_FOLDER.name}.logs"

log_path.parent.mkdir(exist_ok=True, parents=True)
configure_logger(log_path)
logger.info(f"Sample folder: {sample_path}.\nData folder: {DATA_FOLDER}")
logger.info(
    " -------------------------------- STRUCTURED PREP --------------------------------"
)
# %% [markdown]
# # DEFINE A SAMPLE
# In this setup, we require that:
#   - People have a birthday
#   - Minimum 3 tokens (see prep.ipynb)

sample = pl.read_parquet(sample_path / "full_pop_w_birthday.parquet")
PERSON_ID_SUBSAMPLE = sample["person_id"].to_list()
print(len(PERSON_ID_SUBSAMPLE))

# %%
# For use as lower limit for events
df_person_birthday = sample.rename({"birthday": "_birthday"})


# %% [markdown]
# # General data processing
# Your configs should look like this.
#
# DATASET NAME: Final name of file will be FNAME_PREFIX DATASET NAME
#
# dump_path: Path to raw data
#
# date_col_name: Which col to use as ordering column
#
# overall_group_name: String which is preprended to strings to signify group.
#
# columns: A dictionary of column:preprocessing dicts. Columns can be
# - truncated,
# - binned overall,
# - binned grouped by year,
# - prefixed with colname,
# - prefixed with value,
# - kept unchanged.
# Binned columns are assumed to be numeric, but will be returned as strings. All other columns used to create features are assumed to be strings.
#
# Binned columns should not have other args, as this will result in the original col to also be included
#
# EXAMPLE
# ~~~
# dataset_configs = {
#     DATASET NAME: {
#      "dump_path" : PATH TO FILE,
#         "date_col_name": COL TO USE AS DATE,
#         "overall_group_name": SOME_STRING,
#         "columns": {
#             "COL 1": {"truncate": INTEGER WHICH DETERMINES SLICE LENGTH, "prefix_colname": BOOLEAN},
#             "COL 2": {"unchanged": BOOLEAN},
#             "COL 3": {"overall_bin": N BINS}
#             "COL 4": {"yearly_bin": (N BINS, DATE COL)}
#         }
#      },
#     DATASET NAME: {
#         "dump_path" : PATH TO FILE,
#         "date_col_name": COL TO USE AS DATE,
#         "overall_group_name": SOME_STRING,
#         "columns": {
#             "COL 1": {"truncate": INTEGER WHICH DETERMINES SLICE LENGTH, "prefix_colname": BOOLEAN},
#             "COL 2": {"unchanged": BOOLEAN},
#             "COL 3": {"overall_bin": N BINS}
#         }
#     }
# }
# ~~~
#

# %%

IND_many_bins = 100
IND_few_bins = 10

wellbeing_n_bins = 10


dataset_configs = {
    # ---------------------------------- DEMOGRAPHICS ----------------------------------
    # Ok
    # "background": {
    #     "dump_path": FPATH.DUMP_DIR / "all_background_strings.parquet",
    #     "date_col_name": "birthday",
    #     "overall_group_name": "DEM",
    #     "columns": {
    #         "birthyear": {"prefix_colname": True},
    #         "birthmonth": {"prefix_colname": True},
    #         "female": {"prefix_colname": True},
    #         "opr_land": {"prefix_colname": True},
    #         "ie_type": {"prefix_colname": True},
    #         "statsb": {"prefix_colname": True},
    #         "binned_vaegt_barn": {"unchanged": True},
    #         "binned_laengde_barn": {"unchanged": True},
    #         "far_foed_adop": {"prefix_colname": True},
    #         "mor_foed_adop": {"prefix_colname": True},
    #         "n_siblings_at_birth": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "marital_status": {
    #     "dump_path": FPATH.DUMP_DIR / "bef_marital_status.parquet",
    #     "date_col_name": "civ_vfra",
    #     "overall_group_name": "DEM",
    #     "columns": {
    #         "civst": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "migration": {
    #     "dump_path": FPATH.DUMP_DIR / "lifelines_immigration_emmigration.parquet",
    #     "date_col_name": "event_date",
    #     "overall_group_name": "DEM",
    #     "columns": {
    #         "event": {"unchanged": True},
    #     },
    # },
    # # Ok
    # "family_births_deaths": {
    #     "dump_path": FPATH.DUMP_DIR / "family_events.parquet",
    #     "date_col_name": "event_date",
    #     "overall_group_name": "DEM",
    #     "columns": {
    #         "relation": {"prefix_colname": True},
    #         "event": {"unchanged": True},
    #     },
    # },
    # # Ok
    # "address_change": {
    #     "dump_path": FPATH.DUMP_DIR / "befbop_befadr_DUMP.parquet",
    #     "date_col_name": "bop_vfra",
    #     "overall_group_name": "DEM",
    #     "columns": {
    #         "kom": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "bef_updates": {
    #     "dump_path": FPATH.DUMP_DIR / "bef_update.parquet",
    #     "date_col_name": "referencetid",
    #     "overall_group_name": "DEM",
    #     "columns": {
    #         "familie_type": {"prefix_colname": True},
    #         "manual_bin_antboernf": {"prefix_colname": True},
    #         "manual_bin_antpersf": {"prefix_colname": True},
    #         "manual_bin_antefam": {"prefix_colname": True},
    #         "fm_mark": {"prefix_colname": True},
    #         "statsb": {"prefix_colname": True},
    #     },
    # },
    # # ---------------------------------- LABOUR & INCOME ----------------------------------
    # # OK
    # "akm": {
    #     "dump_path": FPATH.DUMP_DIR / "akm_long.parquet",
    #     "date_col_name": "referencetid",
    #     "overall_group_name": "LAB",
    #     "columns": {
    #         "akm_type": {"prefix_colname": True},
    #         "socio13": {"prefix_colname": True},
    #         "socio_gl": {"prefix_colname": True},
    #         "disco08_alle_indk_13": {"prefix_value": "disco08"},
    #         "disco08_loen_indk": {"prefix_value": "disco08"},
    #         "disco08_sel_indk": {"prefix_value": "disco08"},
    #         "disco_alle_indk_13": {"prefix_value": "disco"},
    #         "discoloen_indk": {"prefix_value": "disco"},
    #         "discosel_indk": {"prefix_value": "disco"},
    #         "nace_db07_13": {"prefix_value": "db07"},
    #         "nacea_db07": {"prefix_value": "db07"},
    #         "nacei_db07": {"prefix_value": "db07"},
    #         "nace_13": {"prefix_value": "nace"},
    #         "nacea": {"prefix_value": "nace"},
    #         "nacei": {"prefix_value": "nace"},
    #         "branche_77": {"prefix_value": "branche_77"},
    #         "brchi": {"prefix_value": "branche_77"},
    #         "brchl": {"prefix_value": "branche_77"},
    #     },
    # },
    # Ok
    # "ind": {
    #     "dump_path": FPATH.DUMP_DIR / "ind_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "referencetid",
    #     "overall_group_name": "LAB",
    #     "columns": {
    #         "dispon_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "perindkialt_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "erhvervsindk_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "loenmv_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "honny": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "netovskud_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "off_overforsel_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "dagpenge_kontant_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "arblhumv": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "ovrig_dagpenge_akas_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "kontanthj_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "ovrig_kontanthjalp_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "syg_barsel_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "ovrig_overforsel_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "stip": {"yearly_bin": (IND_few_bins, "referencetid")},
    #         "korstoett": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "korydial": {"yearly_bin": (IND_few_bins, "referencetid")},
    #         "gron_check": {"yearly_bin": (IND_few_bins, "referencetid")},
    #         "offpens_efterlon_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "folkefortid_13": {"yearly_bin": (IND_few_bins, "referencetid")},
    #         "qeftlon": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "privat_pension_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "formueindk_brutto": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "resuink_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "lejev_egen_bolig": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "rentudgpr": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "skatmvialt_13": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "underhol": {"yearly_bin": (IND_many_bins, "referencetid")},
    #         "assets": {"yearly_bin": (IND_many_bins, "referencetid")},
    #     },
    # },
    # "amrun": {
    #     "dump_path": FPATH.DUMP_DIR / "amrun_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "date_to",
    #     "overall_group_name": "LAB",
    #     "columns": {
    #         "soc_status_kode": {"prefix_colname": True},
    #         "disco_kode": {
    #             "prefix_value": "disco08",
    #         },
    #         "arb_hoved_bra_db07": {
    #             "prefix_value": "db07",
    #         },
    #         "bredt_loen_beloeb": {"yearly_bin": (100, "date_to")},
    #         "tilstand_grad_amr": {"overall_bin": 10},
    #         "tilstand_kode_amr_conditional_nullified": {
    #             "prefix_value": "tilstand_kode"
    #         },
    #         "fravaer_besk_kode_nullified": {"prefix_colname": True},
    #         "stoette_besk_kode_nullified": {"prefix_colname": True},
    #         "udd_besk_kode_nullified": {"prefix_colname": True},
    #     },
    # },
    # # ---------------------------------- HEALTH ----------------------------------
    # # Ok
    # "lpr": {
    #     "dump_path": FPATH.DUMP_DIR / "lpr_combined_DUMP_augmented_adaptrunc.parquet",
    #     "date_col_name": "dato_start",
    #     "overall_group_name": "HEA",
    #     "columns": {
    #         "urgency": {"prefix_colname": True},
    #         "patienttype": {"prefix_colname": True},
    #         "aktionsdiagnose_adaptrunc": {"prefix_value": "ICD10"},
    #     },
    # },
    # Ok
    "lmdb": {
        "dump_path": FPATH.DUMP_DIR / "LMDB_DUMP_AUGMENTED.parquet",
        "date_col_name": "eksd",
        "overall_group_name": "HEA",
        "columns": {
            "atc": {"prefix_colname": True},
            "binned_medication_volume": {"unchanged": True},
            # "volume": {"overall_bin": 100},
            # "voltypecode": {"prefix_colname": True},
        },
    },
    # Ok
    # "sygesik": {
    #     "dump_path": FPATH.DUMP_DIR
    #     / "sygesikring_combined_DUMP_AUGMENTED_adaptrunc.parquet",
    #     "date_col_name": "date_col",
    #     "overall_group_name": "HEA",
    #     "columns": {
    #         # "spec2": {"prefix_colname": True},
    #         # "ydelsesnummer": {"prefix_colname": True},
    #         "speciale_adaptrunc": {"prefix_value": "speciale"},
    #     },
    # },
    # # ---------------------------------- EDUCATION ----------------------------------
    # # Ok
    # "primary_school_grades": {
    #     "dump_path": FPATH.DUMP_DIR / "udfk_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "datetime",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "course": {"prefix_colname": True},
    #         "grade": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "high_school_grades": {
    #     "dump_path": FPATH.DUMP_DIR / "karakter_gym_f_DUMP.parquet",
    #     "date_col_name": "date_col",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "course": {"prefix_colname": True},
    #         "course_level": {"prefix_colname": True},
    #         "grade": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "higher_ed_grades": {
    #     "dump_path": FPATH.DUMP_DIR / "karakter_univid_fag_DUMP.parquet",
    #     "date_col_name": "bedommelsesdato",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "grade": {"prefix_colname": True},
    #         "ects": {"overall_bin": 100},
    #     },
    # },
    # Ok
    # "enrollment": {
    #     "dump_path": FPATH.DUMP_DIR / "elev3_enrollment_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "elev3_vfra",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "field": {"prefix_colname": True},
    #         "disced": {"prefix_colname": True},
    #         "udd": {"prefix_colname": True},
    #         "udel": {"prefix_colname": True},
    #         "tilg_art": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "graduation": {
    #     "dump_path": FPATH.DUMP_DIR / "elev3_graduation_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "elev3_vtil",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "field": {"prefix_colname": True},
    #         "disced": {"prefix_colname": True},
    #         "udd": {"prefix_colname": True},
    #         "udel": {"prefix_colname": True},
    #         "audd": {"prefix_value": "audd"},
    #         "afg_art": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "diploma_grades": {
    #     "dump_path": FPATH.DUMP_DIR / "karakter_udd_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "karakter_udd_vtil",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "field": {"prefix_colname": True},
    #         "disced": {"prefix_colname": True},
    #         "audd": {"prefix_value": "audd"},
    #         "grade_7_scale": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "qualifications": {
    #     "dump_path": FPATH.DUMP_DIR / "kvalifikation_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "kvalifikation_vfra",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "field": {"prefix_colname": True},
    #         "disced": {"prefix_colname": True},
    #         "audd": {"prefix_value": "audd"},
    #     },
    # },
    # # Ok
    # "wellbeing": {
    #     "dump_path": FPATH.DUMP_DIR / "wellbeing_patchwork_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "event_date",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "wellbeing_overall_young": {"overall_bin": wellbeing_n_bins},
    #         "wellbeing_overall_old": {"overall_bin": wellbeing_n_bins},
    #         "wellbeing_calm_and_quiet": {"overall_bin": wellbeing_n_bins},
    #         "wellbeing_support_and_inspiration": {"overall_bin": wellbeing_n_bins},
    #         "wellbeing_social": {"overall_bin": wellbeing_n_bins},
    #         "wellbeing_academic": {"overall_bin": wellbeing_n_bins},
    #     },
    # },
    # # Ok
    # "national_tests": {
    #     "dump_path": FPATH.DUMP_DIR / "DNT_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "event_date",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "fag": {"prefix_colname": True},
    #         "bin": {"prefix_value": "dnt_bin"},
    #     },
    # },
    # "absences": {
    #     "dump_path": FPATH.DUMP_DIR / "fravaer_stil_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "event_date",
    #     "overall_group_name": "EDU",
    #     "columns": {
    #         "dageaktiv": {"prefix_colname": True},
    #         "dageialtfra": {"prefix_colname": True},
    #         "dagelovfra": {"prefix_colname": True},
    #         "dagesyg": {"prefix_colname": True},
    #         "dageulovfra": {"prefix_colname": True},
    #     },
    # },
    # # ---------------------------------- SOCIAL CONDITIONS ----------------------------------
    # Ok
    # "decisions": {
    #     "dump_path": FPATH.DUMP_DIR / "kraf_fse_DUMP_AUGMENTED_adaptrunc.parquet",
    #     "date_col_name": "afgoedto",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "bstrflgd": {"overall_bin": 10},
    #         "ubstrflg": {"overall_bin": 10},
    #         "boedeblb": {"yearly_bin": (10, "afgoedto")},
    #         "dagboant": {"overall_bin": True},
    #         "dagbobel": {"yearly_bin": (10, "afgoedto")},
    #         "betbkod": {"prefix_colname": True},
    #         "afgtypko": {"prefix_colname": True},
    #         "frakkod": {"prefix_colname": True},
    #         "frakbkod": {"prefix_colname": True},
    #         "revoke_license_days": {"overall_bin": 10},
    #         "ger7_adaptrunc": {"prefix_value": "ger7"},
    #     },
    # },
    # # Ok
    # "charges": {
    #     "dump_path": FPATH.DUMP_DIR / "krsi_fse_DUMP_AUGMENTED_adaptrunc.parquet",
    #     "date_col_name": "sigtdto",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "charge": {"prefix_colname": True},
    #         "ger7_adaptrunc": {"prefix_value": "ger7"},
    #     },
    # },
    # # Ok
    # "conferred_charges": {
    #     "dump_path": FPATH.DUMP_DIR / "krko_fse_DUMP_AUGMENTED_adaptrunc.parquet",
    #     "date_col_name": "afg_afgoedto",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "afgtypko": {"prefix_colname": True},
    #         "ger7_adaptrunc": {"prefix_value": "ger7"},
    #     },
    # },
    # # Ok
    # "minor_charges": {
    #     "dump_path": FPATH.DUMP_DIR / "krms_DUMP_AUGMENTED_adaptrunc.parquet",
    #     "date_col_name": "sigtdto",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "charge": {"prefix_colname": True},
    #         "ger7_adaptrunc": {"prefix_value": "ger7"},
    #     },
    # },
    # # Ok
    # "victims": {
    #     "dump_path": FPATH.DUMP_DIR / "krof_fse_DUMP_AUGMENTED_adaptrunc.parquet",
    #     "date_col_name": "ofr_gerfradt",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "victim": {"prefix_colname": True},
    #         "ger7_adaptrunc": {"prefix_value": "ger7"},
    #     },
    # },
    # # Ok
    # "jail_parole_start": {
    #     "dump_path": FPATH.DUMP_DIR / "fse_jail_parole_start_adaptrunc.parquet",
    #     "date_col_name": "fgsldto",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "fgslkod": {"prefix_colname": True},
    #         "overfkod": {"prefix_colname": True},
    #         "bstrfkod": {"prefix_colname": True},
    #         "ger7_adaptrunc": {"prefix_value": "ger7"},
    #     },
    # },
    # # Ok
    # "jail_end": {
    #     "dump_path": FPATH.DUMP_DIR / "fse_jail_end.parquet",
    #     "date_col_name": "loesldto",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "loeslkod": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "parole_end": {
    #     "dump_path": FPATH.DUMP_DIR / "fse_parole_end.parquet",
    #     "date_col_name": "prvsldto",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "end_parole": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "out_of_home_placements": {
    #     "dump_path": FPATH.DUMP_DIR / "buaf_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "event_date",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "samtykke": {"prefix_colname": True},
    #         "ansted_klas": {"prefix_colname": True},
    #         "haendelse": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "preventative_measures_start": {
    #     "dump_path": FPATH.DUMP_DIR / "bufo_start_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "sag_vfra",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "start_precaution": {"prefix_colname": True},
    #         "pgf": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "preventative_measures_end": {
    #     "dump_path": FPATH.DUMP_DIR / "bufo_end_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "sag_vtil",
    #     "overall_group_name": "SOC",
    #     "columns": {
    #         "end_precaution": {"prefix_colname": True},
    #         "pgf": {"prefix_colname": True},
    #     },
    # },
    # # Ok
    # "drug_treatment_start": {
    #     "dump_path": FPATH.DUMP_DIR / "smdb_ibib_start_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "ydelsestartdato",
    #     "overall_group_name": "SOC",
    #     "columns": {"start_drug_service": {"prefix_colname": True}},
    # },
    # # Ok
    # "drug_treatment_end": {
    #     "dump_path": FPATH.DUMP_DIR / "smdb_ibib_end_DUMP_AUGMENTED.parquet",
    #     "date_col_name": "ydelseslutdato",
    #     "overall_group_name": "SOC",
    #     "columns": {"end_drug_service": {"prefix_colname": True}},
    # },
}
logger.info("Dataset configurations loaded:")
logger.info(dataset_configs)

# %%
overall_start_time = time.time()
# Iterate over the datasets and process each one
for fname, config in dataset_configs.items():
    check_and_copy_file_or_dir(config["dump_path"])
    start_time = time.time()
    logger.info(f"--- Starting processing for dataset: {fname} ---")
    logger.info("Configuration:\n" + json.dumps(config, indent=2, default=str))

    # Extract specs from columns
    (
        truncation_specs,
        overall_binning_specs,
        yearly_binning_specs,
        cols_to_prefix_colname,
        cols_to_prefix_value,
        cols_unchanged,
    ) = extract_specs(config["columns"])

    logger.info(f"Extracted specs for {fname}:")
    logger.info(
        {
            "truncation_specs": truncation_specs,
            "overall_binning_specs": overall_binning_specs,
            "yearly_binning_specs": yearly_binning_specs,
            "cols_to_prefix_colname": cols_to_prefix_colname,
            "cols_to_prefix_value": cols_to_prefix_value,
            "cols_unchanged": cols_unchanged,
        }
    )

    output_path = DATA_FOLDER / f"{fname}.parquet"
    # Call process_dataset with the extracted specs
    process_dataset(
        dump_path=config["dump_path"],
        person_subsample=PERSON_ID_SUBSAMPLE,
        truncation_specs=truncation_specs,
        cols_to_prefix_colname=cols_to_prefix_colname,
        cols_to_prefix_value=cols_to_prefix_value,
        overall_binning_specs=overall_binning_specs,
        yearly_binning_specs=yearly_binning_specs,
        cols_unchanged=cols_unchanged,
        date_col_name=config["date_col_name"],
        output_file_path=output_path,
        df_remove_pre_date_events=df_person_birthday,
        overall_group_name=config["overall_group_name"],
        chunk_size=CHUNK_SIZE,
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Finished processing {fname}, saved to {output_path}.")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")

overall_elapsed_time = time.time() - overall_start_time
logger.info(f"Overall time taken: {overall_elapsed_time:.2f} seconds")
