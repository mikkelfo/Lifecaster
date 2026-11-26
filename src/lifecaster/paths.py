"""Controls Paths"""

import getpass
import os
import platform
import random
import shutil
from pathlib import Path

from lifecaster.utils import rprint


class ProjectPaths:
    """
    Class containing file paths for the project directory and its subdirectories.
    """

    def __init__(self, project_folder=None) -> None:
        """
        Initialize the ProjectPaths class with base directory.

        Args:
            project_folder (str): The base directory of the project.
        """
        if project_folder is None:
            project_folder = Path(__file__).resolve().parents[2]
            idx = project_folder.parts.index("Lifecaster")
            project_folder = Path(*project_folder.parts[idx - 1 :])

        if platform.system() == "Linux":
            self.NETWORK_DRIVE = Path(
                f"/home/{getpass.getuser()}/kdrev/{project_folder}"
            )
            self.D_DRIVE = Path(
                rf"/mnt/dwork/users/{getpass.getuser()}/{project_folder}"
            )
        else:
            self.NETWORK_DRIVE = Path(rf"K:/{project_folder}")
            self.D_DRIVE = Path(rf"D:/work/{getpass.getuser()}/{project_folder}")

        self.DATA = self.D_DRIVE / "data"
        self.NETWORK_DATA = self.NETWORK_DRIVE / "data"
        self.CHECKPOINTS = self.NETWORK_DRIVE / "checkpoints"
        self.CONFIGS = self.NETWORK_DRIVE / "configs"
        self.TABLES = self.NETWORK_DRIVE / "tables"
        self.FIGURES = self.NETWORK_DRIVE / "figures"
        self.LOGS = self.NETWORK_DRIVE / "logs"
        self.TEMP_FILES = self.DATA / "temp_files"
        self.DUMP_DIR = self.DATA / "dumps"
        self.NETWORK_DUMP_DIR = self.NETWORK_DATA / "dumps"

    def swap_drives(self, fpath: Path):
        """Swaps the drive of fpath by checking parts and replacing with opposite drive"""
        if fpath.parts[: len(self.NETWORK_DRIVE.parts)] == self.NETWORK_DRIVE.parts:
            return self.D_DRIVE / fpath.relative_to(self.NETWORK_DRIVE)
        elif fpath.parts[: len(self.D_DRIVE.parts)] == self.D_DRIVE.parts:
            return self.NETWORK_DRIVE / fpath.relative_to(self.D_DRIVE)
        else:
            raise ValueError("Only supports K and D drive")

    def copy_to_opposite_drive(self, fpath: Path) -> None:
        """
        Copies file or directory to the opposite drive (D->K or K->D).

        Args:
            fpath (Path): The path of the file or directory to copy.
        """
        dest_path = self.swap_drives(fpath)
        if fpath.is_dir():
            shutil.copytree(fpath, dest_path, dirs_exist_ok=True)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, dest_path)


FPATH = ProjectPaths()


def check_and_copy_file_or_dir(file_path: Path, verbosity=3) -> None:
    """
    Checks if a file or directory exists on the specified drive first, then the opposite drive.
    If it exists on the opposite drive, copies it to the specified drive and uses it.
    If the file or directory does not exist on either drive, returns None.

    Args:
        file_path (Path): The path of the file or directory to check and copy if needed.
        verbosity (int): 0 no prints, 1 print only existence, 2 print non-existence, 3 print all
    """
    if file_path.exists():
        if verbosity in [1, 3]:
            rprint(f"{file_path} exists on this drive.")
        return file_path

    swapped_fpath = FPATH.swap_drives(file_path)
    if swapped_fpath.exists():
        if verbosity in [2, 3]:
            rprint(
                f"{file_path} does not exist on this drive. Initiating copy from {swapped_fpath}"
            )
        FPATH.copy_to_opposite_drive(swapped_fpath)
        return file_path
    else:
        if verbosity in [2, 3]:
            rprint(f"{file_path} does not exist on either drive.")
        return None


def get_network_drive():
    if platform.system() == "Linux":
        return f"/home/{getpass.getuser()}/kdrev/"
    else:
        return "K:/"


def get_local_drive():
    if platform.system() == "Linux":
        return rf"/mnt/dwork/users/{getpass.getuser()}"
    else:
        return rf"D:/work/{getpass.getuser()}"


def get_wandb_runid(experiment_name: str):
    """Gets a random name for a run id"""
    # Child ranks started by Lightning inherit this from rank 0
    if run_id := os.environ.get("LIFECASTER_RUN_ID"):
        return run_id
    # fmt: off
    adjectives =['abundant', 'accessible', 'accommodating', 'accurate', 'adaptable', 'adept', 'admirable', 'adorable', 'adventurous', 'affable', 'affectionate', 'aggressive', 'agile', 'agreeable', 'agreeing', 'alert', 'alive', 'alluring', 'amazed', 'amazing', 'amiable', 'amused', 'amusing', 'angry', 'annoyed', 'annoying', 'anxious', 'apprehensive', 'arrogant', 'artful', 'artistic', 'ashamed', 'assertive', 'astonishing', 'astute', 'attentive', 'attractive', 'audacious', 'authentic', 'average', 'avid', 'awful', 'bad', 'balanced', 'beautiful', 'benevolent', 'better', 'bewildered', 'black', 'blissful', 'bloody', 'blue', 'blushing', 'bold', 'bored', 'brainy', 'brave', 'breakable', 'bright', 'brilliant', 'bubbly', 'buoyant', 'busy', 'calm', 'capable', 'carefree', 'careful', 'caring', 'cautious', 'centered', 'charismatic', 'charming', 'cheerful', 'chilly', 'classy', 'clean', 'clear', 'clever', 'cloudy', 'clumsy', 'colorful', 'colossal', 'combative', 'comfortable', 'committed', 'compassionate', 'competent', 'concerned', 'condemned', 'confident', 'confused', 'considerate', 'constructive', 'contemplative', 'content', 'convivial', 'cooperative', 'cordial', 'courageous', 'courteous', 'crafty', 'crazy', 'creative', 'credible', 'creepy', 'crowded', 'cruel', 'curious', 'curvy', 'cute', 'dangerous', 'daring', 'dark', 'dashing', 'dead', 'decisive', 'dedicated', 'defeated', 'defiant', 'deliberate', 'delightful', 'dependable', 'depressed', 'desirable', 'determined', 'devoted', 'different', 'difficult', 'dignified', 'diligent', 'diplomatic', 'direct', 'discerning', 'discreet', 'disgusted', 'distinct', 'disturbed', 'dizzy', 'doubtful', 'drab', 'dull', 'dynamic', 'eager', 'earnest', 'easy', 'effective', 'efficient', 'elated', 'elegant', 'eloquent', 'embarrassed', 'empathetic', 'enchanting', 'encouraging', 'energetic', 'engaged', 'enthusiastic', 'envious', 'ethical', 'evil', 'excited', 'exemplary', 'exotic', 'expensive', 'experienced', 'extraordinary', 'exuberant', 'fair', 'faithful', 'famous', 'fancy', 'fantastic', 'fearless', 'fierce', 'fiery', 'filthy', 'fine', 'focused', 'foolish', 'forthright', 'fragile', 'frail', 'frantic', 'friendly', 'frightened', 'funny', 'gentle', 'genuine', 'gifted', 'giving', 'glamorous', 'gleaming', 'glorious', 'good', 'gorgeous', 'graceful', 'gregarious', 'grieving', 'grotesque', 'grumpy', 'handsome', 'happy', 'hardworking', 'harmonious', 'healthy', 'helpful', 'helpless', 'heroic', 'hilarious', 'homeless', 'homely', 'honest', 'hopeful', 'horrible', 'humble', 'humorous', 'hungry', 'hurt', 'ill', 'imaginative', 'impartial', 'important', 'impossible', 'incomparable', 'independent', 'inexpensive', 'ingenious', 'innocent', 'innovative', 'inquisitive', 'inspired', 'inspiring', 'intellectual', 'intelligent', 'intuitive', 'inventive', 'inviting', 'itchy', 'jealous', 'jittery', 'jolly', 'jovial', 'joyful', 'joyous', 'judicious', 'just', 'keen', 'kind', 'knowledgeable', 'laudable', 'lazy', 'leaderly', 'lenient', 'light', 'likable', 'lively', 'logical', 'lonely', 'long', 'lovely', 'loyal', 'lucid', 'lucky', 'magical', 'magnificent', 'majestic', 'mature', 'methodical', 'meticulous', 'mighty', 'mindful', 'misty', 'modern', 'modest', 'motionless', 'motivated', 'muddy', 'mushy', 'mysterious', 'nasty', 'natural', 'naughty', 'nervous', 'nice', 'nifty', 'noble', 'nurturing', 'nutty', 'obedient', 'obnoxious', 'odd', 'open', 'optimistic', 'orderly', 'organized', 'outgoing', 'outrageous', 'outstanding', 'panicky', 'passionate', 'patient', 'peaceful', 'perceptive', 'perfect', 'persistent', 'philosophical', 'plain', 'playful', 'pleasant', 'poised', 'polished', 'poor', 'positive', 'powerful', 'practical', 'precious', 'precise', 'prickly', 'proactive', 'productive', 'proficient', 'proud', 'prudent', 'punctual', 'putrid', 'puzzled', 'quaint', 'quick', 'radiant', 'rational', 'real', 'reliable', 'relieved', 'remarkable', 'repulsive', 'resilient', 'resolute', 'resourceful', 'respectful', 'responsible', 'responsive', 'reverent', 'rich', 'robust', 'savvy', 'scary', 'scholarly', 'selfish', 'selfless', 'sensible', 'serene', 'sharp', 'shiny', 'shrewd', 'shy', 'silly', 'sincere', 'skillful', 'sleepy', 'smart', 'smiling', 'smoggy', 'sociable', 'solid', 'sore', 'sparkling', 'spirited', 'splendid', 'spontaneous', 'spotless', 'steadfast', 'steady', 'stormy', 'strange', 'strategic', 'strong', 'studious', 'stupid', 'successful', 'super', 'supportive', 'sustainable', 'sympathetic', 'tactful', 'talented', 'tame', 'tasty', 'tenacious', 'tender', 'tense', 'terrible', 'thankful', 'thoughtful', 'thoughtless', 'thriving', 'tired', 'tireless', 'tolerant', 'tough', 'troubled', 'trustworthy', 'ugliest', 'ugly', 'unassuming', 'unbiased', 'understanding', 'uninterested', 'unique', 'unsightly', 'unusual', 'upbeat', 'upset', 'uptight', 'vast', 'versatile', 'vibrant', 'victorious', 'vigilant', 'vivacious', 'vivid', 'wandering', 'warm', 'weary', 'wicked', 'wild', 'wise', 'witty', 'worried', 'worrisome', 'wrong', 'youthful', 'zany', 'zealous', 'zestful']
    nouns = ["tiger", "falcon", "whale", "dragon", "phoenix", "eagle", "panther", "lion", "wolf", "hawk", "unicorn", "griffin", "centaur", "hydra", "lynx", "serpent", "kraken", "bear", "cheetah", "orca", "rhino", "cougar", "leopard", "cobra", "jaguar", "raven", "sparrow", "sphinx", "chameleon", "pelican", "giraffe", "fox", "ocelot", "mustang", "buffalo", "stallion", "hippo", "bison", "shark", "gazelle", "otter", "penguin"]
    # fmt: on

    file_dir = FPATH.LOGS / experiment_name

    run_number = 1
    if len(files := list(file_dir.glob("*"))) > 0:
        run_number = max([int(file.stem.split("_")[0]) for file in files])
        run_number += 1
    run_id = f"{run_number:03d}_{random.choice(adjectives)}_{random.choice(nouns)}"
    os.environ["LIFECASTER_RUN_ID"] = run_id

    return run_id
