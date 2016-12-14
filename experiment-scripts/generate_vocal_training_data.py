"""Script to generate vocal training data"""
from __future__ import print_function

import argparse
import glob
import medleydb as mdb
from medleydb import mix
import os
import shutil
import sox


def make_training_set(output_dir):
    annotations_dir = os.path.join(output_dir, 'annotations')
    audio_dir = os.path.join(output_dir, 'audio')

    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)
    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)

    mtracks = mdb.load_melody_multitracks()
    for mtrack in mtracks:
        print(mtrack.track_id)

        if mtrack.predominant_stem.instrument not in mix.VOCALS:
            print("no vocals...skipping")
            print("")
            continue

        output_path = os.path.join(
            audio_dir, '{}_vocal.wav'.format(mtrack.track_id)
        )
        print([(i, v.instrument) for i, v in mtrack.stems.items()])

        need_remix = False
        stem_indices = mtrack.stems.keys()
        stem_indices.remove(mtrack.predominant_stem.stem_idx)
        for i in stem_indices:
            if mtrack.stems[i].instrument in mix.VOCALS:
                need_remix = True

        if need_remix:
            print("needs remix")
            mix_stems = [
                i for i, v in mtrack.stems.items()
                if v.instrument not in mix.VOCALS
            ]
            mix_stems.append(mtrack.predominant_stem.stem_idx)
            mix.mix_multitrack(mtrack, output_path, stem_indices=mix_stems)
        else:
            print("using orignal mix")
            shutil.copy(mtrack.mix_path, output_path)

        annotation_fpath = os.path.join(
            annotations_dir, '{}_vocal.csv'.format(mtrack.track_id)
        )
        shutil.copy(mtrack.melody1_fpath, annotation_fpath)
        print("")


def make_testing_set(audio_path, tony_path, annot_path, tony_script):
    for audio_fpath in glob.glob(os.path.join(audio_path, '*.wav')):
        fname = os.path.basename(audio_fpath).split('.')[0]
        tony_fpath = os.path.join(tony_path, "{}.csv".format(fname))
        output_fpath = os.path.join(annot_path, "{}.csv".format(fname))
        duration = sox.file_info.duration(audio_fpath)
        os.system('python {} {} {} {}'.format(
            tony_script, tony_fpath, output_fpath, duration)
        )


def main(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    train_data_path = os.path.join(args.output_path, 'train-mdb')
    test_data_path = os.path.join(args.output_path, 'test-world-music')

    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)

    if not os.path.exists(test_data_path):
        os.mkdir(test_data_path)

    make_training_set(train_data_path)
    make_testing_set(
        args.test_audio_path, args.test_tony_path,
        test_data_path, args.tony_path
    )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Fill in missing time frames in a Tony file,"
                    "write the output to file.")
    PARSER.add_argument("output_path",
                        type=str,
                        help="Name of folder to save output.")
    PARSER.add_argument("test_audio_path",
                        type=str,
                        help="Path to test data audio files.")
    PARSER.add_argument("test_tony_path",
                        type=str,
                        help="Path to test data tony annotation files.")
    PARSER.add_argument("tony_script",
                        type=str,
                        help="Path to fill_tony_file script.")
    main(PARSER.parse_args())

