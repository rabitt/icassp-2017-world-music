from __future__ import print_function

import argparse
import glob
import json
import matplotlib.pyplot as plt
import motif
import numpy as np
import os


def build_file_pairs(train_audio_path, train_annotations_path,
                     test_audio_path, test_annotations_path):
    train_pairs = []
    for audio_fpath in glob.glob(os.path.join(train_audio_path, '*.wav')):
        annot_fpath = os.path.join(
            train_annotations_path,
            os.path.basename(audio_fpath).split('.')[0] + '.csv'
        )
        train_pairs.append((audio_fpath, annot_fpath))

    test_pairs = []
    for audio_fpath in glob.glob(os.path.join(test_audio_path, '*.wav')):
        annot_fpath = os.path.join(
            test_annotations_path,
            os.path.basename(audio_fpath).split('.')[0] + '.csv'
        )
        test_pairs.append((audio_fpath, annot_fpath))

    return train_pairs, test_pairs


def score_clf(contour_classifier, X, Y):
    Y_pred = contour_classifier.predict(X)
    scores = contour_classifier.score(Y_pred, Y)
    return scores


def plot_feature_importances(contour_classifier, feature_extractor, output_dir):
    plt.figure(figsize=(17, 5))
    x_vals = range(len(contour_classifier.clf.feature_importances_))
    plt.bar(x_vals, contour_classifier.clf.feature_importances_)
    plt.xticks(
        [v + 0.4 for v in x_vals],
        feature_extractor.feature_names,
        rotation='vertical'
    )
    output_path = os.path.join(output_dir, 'feature_importances.pdf')
    plt.savefig(output_path, ext='pdf', bbox_inches='tight')


def compute_contour_coverage(contour_list, file_pairs):
    extractor_recall = {}
    for ctr, pair in zip(contour_list, file_pairs):
        annotation = pair[1]
        coverage_score = ctr.coverage(annotation)
        extractor_recall[ctr.audio_filepath] = coverage_score['Recall']
    return extractor_recall


def predict_on_world_music(world_music_dir, output_dir, contour_extractor,
                           feature_extractor, contour_classifier):
    prediction_set = glob.glob(os.path.join(world_music_dir, 'audio', '*.wav'))
    contour_output_dir = os.path.join(output_dir, 'world_music_contours')
    if not os.path.exists(contour_output_dir):
        os.mkdir(contour_output_dir)

    save_files = []
    for audio_filepath in prediction_set:
        save_name = os.path.basename(audio_filepath).split('.')[0]
        save_path = os.path.join(
            contour_output_dir, "{}.csv".format(save_name)
        )
        print(save_name)

        ctr = contour_extractor.compute_contours(audio_filepath)

        X = feature_extractor.compute_all(ctr)
        Y = contour_classifier.predict(X)

        save_nums = [
            n for n in ctr.nums if Y[n] >= contour_classifier.threshold
        ]
        ctr.save_contours_subset(save_path, save_nums)
        save_files.append(save_path)

    return save_files


def get_contour_stats(contour_extractor, prediction_files):
    n_contours = []
    min_duration = []
    max_duration = []
    avg_duration = []
    max_idx = []
    n_no_contours = 0
    for save_path in prediction_files:
        print(save_path)
        i, t, f, s = contour_extractor._load_contours(save_path)

        if len(i) == 0:
            n_no_contours += 1
            continue

        cnums = list(set(i))
        n_contours.append(len(cnums))
        max_idx.append(np.max(cnums))
        c_durations = []
        for n in cnums:
            idx = np.where(i == n)[0]
            c_durations.append(np.max(t[idx]) - np.min(t[idx]))
        min_duration.append(np.min(c_durations))
        max_duration.append(np.max(c_durations))
        avg_duration.append(np.mean(c_durations))

    stats = {
        'average n_contours': np.mean(n_contours),
        'std n_contours': np.std(n_contours),
        'max n_contours': np.max(n_contours),
        'shortest contour': np.min(min_duration),
        'longest contour': np.max(max_duration),
        'average contour length': np.mean(avg_duration),
        'n_tracks with no contours': n_no_contours
    }
    return stats

def main(args):
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print("=" * 80)
    print("Building file pairs...")
    print("=" * 80)
    train_pairs, test_pairs = build_file_pairs(
        args.train_audio_path, args.train_annotations_path,
        args.test_audio_path, args.test_annotations_path
    )

    # initialize feature contour extractor, feature extractor, and classifier
    contour_extractor = motif.run.get_extract_module('salamon')
    feature_extractor = motif.run.get_features_module('bitteli')
    contour_classifier = motif.run.get_classify_module('random_forest')

    # build train/test matrices
    print("=" * 80)
    print("Building training and testing data...")
    print("=" * 80)
    X_train, Y_train, train_contours = motif.run.process_with_labels(
        contour_extractor, feature_extractor, train_pairs
    )
    X_test, Y_test, test_contours = motif.run.process_with_labels(
        contour_extractor, feature_extractor, test_pairs
    )

    # compute coverage
    try:
        train_coverage = compute_contour_coverage(train_contours, train_pairs)
        test_coverage = compute_contour_coverage(test_contours, test_pairs)
        coverage = {
            'train': {
                'mean': np.mean(train_coverage.values()),
                'std': np.std(train_coverage.values()),
                'min': np.min(train_coverage.values()),
                'max': np.max(train_coverage.values())
            },
            'test': {
                'mean': np.mean(test_coverage.values()),
                'std': np.std(test_coverage.values()),
                'min': np.min(test_coverage.values()),
                'max': np.max(test_coverage.values())
            },
        }
        print("=" * 80)
        print("Contour extractor coverage:")
        print("=" * 80)
        print(coverage)

        # save coverage to file
        overall_coverage_path = os.path.join(output_path, 'overall_coverage.json')
        train_coverage_path = os.path.join(output_path, 'train_coverage.json')
        test_coverage_path = os.path.join(output_path, 'test_coverage.json')
        with open(overall_coverage_path, 'w') as fhandle:
            json.dump(coverage, fhandle)
        with open(train_coverage_path, 'w') as fhandle:
            json.dump(train_coverage, fhandle)
        with open(test_coverage_path, 'w') as fhandle:
            json.dump(test_coverage, fhandle)
    except:
        print("[Warning] Coverage computation failed")

    # fit model
    print("=" * 80)
    print("Training classifier...")
    print("=" * 80)
    contour_classifier.fit(X_train, Y_train)

    # compute training and testing scores
    print("=" * 80)
    print("Scoring classifier...")
    print("=" * 80)
    train_score = score_clf(contour_classifier, X_train, Y_train)
    test_score = score_clf(contour_classifier, X_test, Y_test)
    score_dict = {"train": train_score, "test": test_score}
    print(score_dict)

    # save scores to file
    try:
        scores_path = os.path.join(output_path, 'classifier_scores.json')
        with open(scores_path, 'w') as fhandle:
            json.dump(score_dict, fhandle)
    except:
        print("[Warning] Saving classifier scores failed")

    try:
        print("=" * 80)
        print("Creating feature importances plot...")
        print("=" * 80)
        plot_feature_importances(
            contour_classifier, feature_extractor, output_path
        )
    except:
        print("[Warning] Creating feature importances plot failed")

    print("=" * 80)
    print("Computing contours for world music prediction set...")
    print("=" * 80)
    prediction_files = predict_on_world_music(
        args.prediction_audio_path, output_path, contour_extractor,
        feature_extractor, contour_classifier
    )

    try:
        print("=" * 80)
        print("Computing contour statistics for world music prediction set...")
        print("=" * 80)
        vocal_contour_stats = get_contour_stats(
            contour_extractor, prediction_files
        )
        contour_stats_path = os.path.join(
            output_path, 'world_music_contour_stats.json'
        )
        with open(contour_stats_path, 'w') as fhandle:
            json.dump(vocal_contour_stats, fhandle)
    except:
        print("[Warning] Computing contour stats failed")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Fill in missing time frames in a Tony file,"
                    "write the output to file.")
    PARSER.add_argument("train_audio_path",
                        type=str,
                        help="Path to training audio files.")
    PARSER.add_argument("train_annotations_path",
                        type=str,
                        help="Path to training annotation files.")
    PARSER.add_argument("test_audio_path",
                        type=str,
                        help="Path to testing audio files.")
    PARSER.add_argument("test_annotations_path",
                        type=str,
                        help="Path to testing annotation files.")
    PARSER.add_argument("prediction_audio_path",
                        type=str,
                        help="Path to prediction audio files.")
    PARSER.add_argument("output_path",
                        type=str,
                        help="Path to save output data and plots.")
    main(PARSER.parse_args())
