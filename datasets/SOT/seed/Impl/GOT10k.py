import os
from datasets.types.data_split import DataSplit
from datasets.SOT.constructor.base_interface import SingleObjectTrackingDatasetConstructor
from miscellanies.parser.ini import parse_ini_file
from miscellanies.parser.txt import load_numpy_array_from_txt
from miscellanies.numpy.dtype import try_get_int_array
import ast
import numpy as np

_category_names = [
    'JetLev-Flyer', 'abrocome', 'addax', 'african elephant', 'agama',
    'aircraft', 'aircraft carrier', 'airplane', 'airship',
    'alaskan brown bear', 'albatross', 'alligator', 'alligator lizard',
    'alpaca', 'ambulance', 'american bison', 'amphibian', 'angora cat',
    'angora goat', 'anteater', 'antelope', 'antelope head', 'aoudad',
    'appaloosa', 'arabian camel', 'armadillo', 'armored personnel carrier',
    'asian crocodile', 'asiatic black bear', 'assault gun',
    'australian blacksnake', 'australian terrier', 'autogiro',
    'bactrian camel', 'badger', 'balance car', 'bald eagle', 'balloon',
    'banded gecko', 'banded palm civet', 'banded sand snake', 'barge',
    'barracuda', 'barrow', 'basilisk', 'basketball', 'bathyscaphe',
    'beaded lizard', 'bear cub', 'bee', 'beetle', 'belgian hare', 'berlin',
    'bernese mountain dog', 'bettong', 'bezoar goat', 'bhand truck',
    'bicycle wheel', 'big truck', 'bighorn', 'binturong', 'black rhinoceros',
    'black squirrel', 'black stork', 'black vulture',
    'black-crowned night heron', 'black-necked cobra', 'blindworm blindworm',
    'blue point siamese', 'boa', 'boar', 'boneshaker', 'border collie',
    'border terrier', 'borzoi', 'bovine', 'brahman', 'brambling', 'brig',
    'brigantine', 'broadtail', 'brougham', 'brush-tailed porcupine',
    'bulldozer', 'bumboat', 'bumper car', 'burmese cat', 'buzzard',
    'cabbageworm', 'camel head', 'canada porcupine', 'canal boat', 'canoeing',
    'cape buffalo', 'capital ship', 'capybara', 'car', 'car wheel', 'carabao',
    'cargo ship', 'cashmere goat', 'cayuse', 'chameleon', 'checkered whiptail',
    'cheetah', 'cheviot', 'chevrotain', 'chiacoan peccary', 'chimaera',
    'chimpanzee', 'chinese paddlefish', 'cinnamon bear', 'clam', 'coati',
    'cockroach', 'collared lizard', 'collared peccary', 'common kingsnake',
    'common starling', 'common wallaby', 'common zebra', 'compact', 'condor',
    'convertible', 'corn borer', 'cornetfish', 'corvette', 'cotswold', 'coupe',
    'covered wagon', 'cow pony', 'coypu', 'crab', 'crayfish', 'crow',
    'cruise missile', 'cryptoprocta', 'ctenophore', 'curassow', 'cutter',
    'cypriniform fish cypriniform', 'dall sheep', 'daphnia', 'deer',
    'destroyer escort', 'dhow', 'diesel locomotive', 'dogsled',
    'domestic llama', 'dove', 'dragon-lion dance', 'drone', 'duck', 'dumpcart',
    'dune buggy', 'earwig', 'eastern grey squirrel', 'egyptian cat',
    'elasmobranch', 'electric', 'electric locomotive', 'elk', 'european hare',
    'eye', 'face', 'fairy shrimp', 'fall cankerworm', 'false scorpion',
    'fanaloka', 'fire engine', 'fireboat', 'flamingo', 'football',
    'forest goat', 'forklift', 'fox squirrel', 'frilled lizard',
    'garden centipede', 'gavial', 'gemsbok', 'genet', 'giant armadillo',
    'giant kangaroo', 'giant panda', 'gila monster', 'giraffe', 'glider',
    'gnu', 'go-kart', 'golden hamster', 'golfcart', 'goose', 'gopher', 'goral',
    'gorilla', 'greater pichiciego', 'green lizard', 'green snake',
    "grevy's zebra", 'grizzly', 'grouse', 'guanaco', 'guard ship', 'gunboat',
    'hagfish', 'half track', 'halfbeak', 'hammerhead', 'hand', 'hand truck',
    'hazel mouse', 'hedge sparrow', 'helicopter', 'hermit crab', 'heron',
    'hinny', 'hippopotamus', 'hog-nosed skunk', 'horse cart',
    'horseless carriage', 'horseshoe crab', 'hudson bay collared lemming',
    'humvee', 'hyrax', 'ibex', 'ice bear', 'iceboat', 'icebreaker',
    'ichneumon', 'indian cobra', 'indian mongoose', 'indian rat snake',
    'indian rhinoceros', 'interceptor', 'irish terrier', 'irish wolfhound',
    'italian greyhound', 'jaguar', 'japanese deer', 'jeep', 'jellyfish',
    'jinrikisha', 'junk', 'kanchil', 'kinkajou', 'kitty', 'koala', 'lamprey',
    'lander', 'landing craft', 'langur', 'lappet caterpillar', 'large civet',
    'legless lizard', 'lemur', 'leopard', 'lerot', 'lesser kudu',
    'lesser panda', 'lion', 'lippizan', 'long-tailed porcupine', 'longwool',
    'lorry', 'louse', 'lovebird', 'luge', 'lugger', 'macaque', 'magpie',
    'mailboat', 'malayan tapir', 'manatee', 'mangabey', 'manx',
    'marco polo sheep', 'marine iguana', 'marmoset', 'mealworm', 'medusa',
    'merino', 'millipede', 'minicab', 'minisub', 'minivan', 'mole', 'moloch',
    'morgan', 'moth', 'motor scooter', 'motorboat', 'motorcycle',
    'motorcycle wheel', 'mouflon', 'mountain beaver', 'mountain bike',
    'mountain chinchilla', 'mountain goat', 'mountain skink', 'mountain zebra',
    'mule', 'multistage rocket', 'musk ox', 'night snake', 'nilgai',
    'northern snakehead', 'opossum rat', 'orangutan', 'orthopter', 'osprey',
    'ostrich', 'otter', 'otter shrew', 'otterhound', 'owl', 'ox', 'oxcart',
    'pace car', 'pacific walrus', 'paddlefish', 'pademelon', 'palomino',
    'panzer', 'passenger ship', 'patrol boat', 'peba', 'pedicab', 'pekinese',
    'pelican', 'peludo', 'penguin', "pere david's deer", 'person',
    'person head', 'pheasant', 'pickup truck', 'pine marten', 'pine snake',
    'pink cockatoo', 'pinto', 'platypus', 'plodder', 'plover', 'polar hare',
    'pole horse', 'polo pony', 'pony cart', 'pouched mole', 'prawn',
    'praying mantis', 'proboscis monkey', "przewalski's horse", 'pt boat',
    'pung', 'punt', 'push-bike', 'putterer', 'pygmy mouse', 'quarter horse',
    'raccoon', 'racerunner', 'rambouillet', 'raven', 'reconnaissance plane',
    'reconnaissance vehicle', 'red squirrel', 'rhodesian ridgeback',
    'road race', 'roadster', 'rock hyrax', 'roller coaster', 'rolling stock',
    'round-tailed muskrat', 'sailboard', 'sailboat', 'sassaby', 'scooter',
    'scorpion', 'scotch terrier', 'sea otter', 'seahorse', 'seal',
    'sealyham terrier', 'serow', 'shoe', 'shopping cart', 'shrimp',
    'shrimpfish', 'skateboard', 'skibob', 'skidder', 'skunk', 'sloth',
    'sloth bear', 'small boat', 'snail', 'snow leopard', 'snowmobil',
    'snowplow', 'soccer ball', 'sonoran whipsnake', 'sow', 'space shuttle',
    'spider', 'spider monkey', 'sport utility', 'sports car', 'spotted skunk',
    'squirrel monkey', 'standard poodle', 'standard schnauzer',
    'stanley steamer', 'stealth bomber', 'stealth fighter', 'steam locomotive',
    'steamboat', 'steamroller', 'stickleback', 'striped skunk', 'subcompact',
    'suricate', 'swallow', 'swamprabbit', 'tabby', 'tadpole shrimp', 'takin',
    'tank', 'tank destroyer', 'tasmanian devil', 'termite',
    'texas horned lizard', 'tiger cat', 'tiglon', 'tobacco hornworm',
    'toboggan', 'tortoiseshell', 'traffic sign', 'train', 'tramcar', 'trawler',
    'tree lizard', 'tree shrew', 'tricycle wheel', 'trilobite', 'trolleybus',
    'troopship', 'truck', 'turtle', 'unicycle', 'urial', 'virginia deer',
    'viscacha', 'volleyball', 'warship', 'warthog', 'wasp', 'water cart',
    'water chevrotain', 'water wagon', 'water-drop', 'whale', 'wheelchair',
    'white elephant', 'white rhinoceros', 'white stork',
    'white-tailed jackrabbit', 'whitetail prairie dog', 'wildboar', 'wildcat',
    'wisent', 'wolverine', 'wombat', 'woodcock', 'woodlouse',
    'woolly bear moth', 'woolly monkey', 'worm lizard', 'yacht',
    'yellow-throated marten', 'zebra-tailed lizard'
]


def _construct_GOT10k_public_data(
        constructor: SingleObjectTrackingDatasetConstructor, sequence_list,
        category_name_id_map):
    for sequence_name, sequence_path in sequence_list:
        images = os.listdir(sequence_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        absence_array = load_numpy_array_from_txt(os.path.join(
            sequence_path, 'absence.label'),
                                                  dtype=bool)
        # Values 0~8 in file cover.label correspond to ranges of object visible ratios: 0%, (0%, 15%], (15%~30%], (30%, 45%], (45%, 60%], (60%, 75%], (75%, 90%], (90%, 100%) and 100% respectively.
        cover_array = load_numpy_array_from_txt(os.path.join(
            sequence_path, 'cover.label'),
                                                dtype=int)
        cut_by_image_array = load_numpy_array_from_txt(os.path.join(
            sequence_path, 'cut_by_image.label'),
                                                       dtype=bool)

        bounding_boxes = load_numpy_array_from_txt(os.path.join(
            sequence_path, 'groundtruth.txt'),
                                                   delimiter=',')
        bounding_boxes = try_get_int_array(bounding_boxes)

        assert len(images) == len(absence_array) == len(cover_array) == len(
            cut_by_image_array) == len(bounding_boxes)
        meta_info = parse_ini_file(os.path.join(sequence_path,
                                                'meta_info.ini'))['METAINFO']

        object_class = meta_info['object_class']
        frame_size = meta_info['resolution']
        frame_size = ast.literal_eval(frame_size)
        fps = meta_info['anno_fps']
        assert fps.endswith('Hz')
        fps = int(fps[:-2])

        with constructor.new_sequence(
                category_name_id_map[object_class]) as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            sequence_constructor.merge_attributes(meta_info)
            sequence_constructor.set_fps(fps)
            for image, bounding_box, absence, cover, cut_by_image in zip(
                    images, bounding_boxes, absence_array, cover_array,
                    cut_by_image_array):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(
                        os.path.join(sequence_path, image), frame_size)
                    frame_constructor.set_bounding_box(bounding_box.tolist(),
                                                       validity=not absence)
                    frame_constructor.set_object_attribute(
                        'absence', absence.item())
                    frame_constructor.set_object_attribute(
                        'cover', cover.item())
                    frame_constructor.set_object_attribute(
                        'cut_by_image', cut_by_image.item())


def _construct_GOT10k_non_public_data(
        constructor: SingleObjectTrackingDatasetConstructor, sequence_list):
    for sequence_name, sequence_path in sequence_list:
        images = os.listdir(sequence_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        bounding_box = load_numpy_array_from_txt(os.path.join(
            sequence_path, 'groundtruth.txt'),
                                                 delimiter=',')
        bounding_box = try_get_int_array(bounding_box)

        assert bounding_box.ndim == 1 and bounding_box.shape[0] == 4

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for index_of_image, image in enumerate(images):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(
                        os.path.join(sequence_path, image))
                    if index_of_image == 0:
                        frame_constructor.set_bounding_box(
                            bounding_box.tolist())


def construct_GOT10k(constructor: SingleObjectTrackingDatasetConstructor,
                     seed):
    root_path = seed.root_path
    data_split = seed.data_split
    sequence_filter = seed.sequence_filter

    if data_split == DataSplit.Training:
        folder = 'train'
    elif data_split == DataSplit.Validation:
        folder = 'val'
    elif data_split == DataSplit.Testing:
        folder = 'test'
    else:
        raise RuntimeError(f'Unsupported dataset split {data_split}')

    constructor.set_category_id_name_map(
        {k: v
         for k, v in enumerate(_category_names)})

    sequence_list = []
    for sequence_name in open(os.path.join(root_path, folder, 'list.txt'),
                              'r'):
        sequence_name = sequence_name.strip()
        current_sequence_path = os.path.join(root_path, folder, sequence_name)
        sequence_list.append((sequence_name, current_sequence_path))

    if sequence_filter is not None:
        sequence_id_file_path = os.path.join(os.path.dirname(__file__),
                                             'data_specs',
                                             f'{sequence_filter}.txt')
        sequence_ids = np.loadtxt(sequence_id_file_path, dtype=np.uint32)
        sequence_list = [sequence_list[id_] for id_ in sequence_ids]

    constructor.set_total_number_of_sequences(len(sequence_list))

    if data_split in (DataSplit.Training, DataSplit.Validation):
        _construct_GOT10k_public_data(
            constructor, sequence_list,
            {v: k
             for k, v in enumerate(_category_names)})
    else:
        _construct_GOT10k_non_public_data(constructor, sequence_list)
