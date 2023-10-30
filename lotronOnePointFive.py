import lanceotron
import lotron2
import csv
import pandas as pd
import numpy as np
import pyBigWig
import pickle
from sklearn.preprocessing import StandardScaler
import natsort
from tensorflow import keras
import tensorflow.keras.backend as K


def find_candidate_peaks_chrom(threshold_cumulative_init, chrom, background_list, window_list, threshold_list, min_size, max_size, background_global_min=None):
    candidate_peaks_df = lotron2.call_candidate_peaks_lotron_chrom(threshold_cumulative_init, chrom, background_list, window_list, threshold_list, min_size, max_size, background_global_min=background_global_min)
    return candidate_peaks_df


def find_candidate_peaks_genome(threshold_cumulative_init, background_list, window_list, threshold_list, min_size, max_size, include_special_chromosomes=False, background_global_min=None):
    candidate_peaks_df = call_candidate_peaks_lotron_genome(threshold_cumulative_init, background_list, window_list, threshold_list, min_size, max_size, include_special_chromosomes=include_special_chromosomes, background_global_min=background_global_min)
    return candidate_peaks_df


# standard_scaler_wide_v5_03.p and standard_scaler_deep_v5_03.p are default scalers for Lanceotron 1
# weights files can be downloaded from https://github.com/LHentges/LanceOtron/tree/master/lanceotron/lanceotron/static

def scale_wide_and_deep_arrays(wide_array, deep_array, wide_array_weights='standard_scaler_wide_v5_03.p', deep_array_weights='standard_scaler_deep_v5_03.p'):
    standard_scaler_wide = pickle.load(open(wide_array_weights, 'rb'))
    wide_array_norm = standard_scaler_wide.transform(wide_array)
    wide_array_norm = np.expand_dims(wide_array_norm, axis=2)
    standard_scaler = StandardScaler()
    deep_array_norm_T = standard_scaler.fit_transform(deep_array.T)
    standard_scaler_deep = pickle.load(open(deep_array_weights, 'rb'))
    deep_array_norm = standard_scaler_deep.transform(deep_array_norm_T.T)
    deep_array_norm = np.expand_dims(deep_array_norm, axis=2)
    return wide_array_norm, deep_array_norm


# wide_and_deep_fully_trained_v5_03.h5 is default model for Lanceotron 1
# weights files can be downloaded from https://github.com/LHentges/LanceOtron/tree/master/lanceotron/lanceotron/static

def build_model(model_weights_file='wide_and_deep_fully_trained_v5_03.h5'):

    deep_dense_size = 10
    dropout_rate = 0.5
    first_filter_num = 70
    first_filter_size = 9
    hidden_filter_num = 120
    hidden_filter_size = 6
    learning_rate = 0.0001
    wide_and_deep_dense_size = 70

    input_wide = keras.layers.Input(shape=(12, 1))
    wide_model = keras.layers.Flatten()(input_wide)
    input_deep = keras.layers.Input((2000, 1))
    deep_model = input_deep

    # deep model first conv layer
    deep_model = keras.layers.Convolution1D(
        first_filter_num, kernel_size=first_filter_size, padding="same"
    )(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)

    # deep model - 4 conv blocks
    deep_model = keras.layers.Convolution1D(
        hidden_filter_num, kernel_size=hidden_filter_size, padding="same"
    )(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(
        hidden_filter_num, kernel_size=hidden_filter_size, padding="same"
    )(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(
        hidden_filter_num, kernel_size=hidden_filter_size, padding="same"
    )(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(
        hidden_filter_num, kernel_size=hidden_filter_size, padding="same"
    )(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    # deep model - dense layer with dropout
    deep_model = keras.layers.Dense(deep_dense_size)(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.Dropout(dropout_rate)(deep_model)
    deep_model = keras.layers.Flatten()(deep_model)

    # shape output only dense layer
    shape_output = keras.layers.Dense(
        2, activation="softmax", name="shape_classification"
    )(deep_model)

    # p-value output only dense layer
    pvalue_output = keras.layers.Dense(
        2, activation="softmax", name="pvalue_classification"
    )(wide_model)

    # combine wide and deep paths
    concat = keras.layers.concatenate([wide_model, deep_model, pvalue_output])
    wide_and_deep = keras.layers.Dense(wide_and_deep_dense_size)(concat)
    wide_and_deep = keras.layers.BatchNormalization()(wide_and_deep)
    wide_and_deep = keras.layers.LeakyReLU()(wide_and_deep)
    wide_and_deep = keras.layers.Dense(wide_and_deep_dense_size)(wide_and_deep)
    wide_and_deep = keras.layers.BatchNormalization()(wide_and_deep)
    wide_and_deep = keras.layers.LeakyReLU()(wide_and_deep)
    output = keras.layers.Dense(2, activation="softmax", name="overall_classification")(
        wide_and_deep
    )
    model = keras.models.Model(
        inputs=[input_deep, input_wide], outputs=[output, shape_output, pvalue_output]
    )

    # load model weights
    model.load_weights(model_weights_file)
    return model


def score_peaks_cov_array(chrom_cov_array, coord_list, read_coverage_rphm, wide_array_weights='standard_scaler_wide_v5_03.p', deep_array_weights='standard_scaler_deep_v5_03.p', model_weights_file='wide_and_deep_fully_trained_v5_03.h5'):
    # N.B lanceotron.extract_signal_wide_and_deep_chrom expects the cov_array to be in units of reads per hundred million  
    wide_array, deep_array = lanceotron.extract_signal_wide_and_deep_chrom(chrom_cov_array/read_coverage_rphm, coord_list, read_coverage_rphm)
    wide_array_norm, deep_array_norm = scale_wide_and_deep_arrays(wide_array, deep_array, wide_array_weights, deep_array_weights)
    model = build_model(model_weights_file)
    model_classifications = model.predict([deep_array_norm, wide_array_norm], verbose=0)
    model_class_array = np.array(model_classifications)
    K.clear_session()
    out_df = pd.DataFrame(coord_list, columns=['Start', 'End'])
    #print(model_classifications.shape)
    #print(model_classifications[:5])
    #print(out_df.head())
    out_df[['Overall_peak_score', 'Shape_score', 'Enrichment_score']] = pd.DataFrame(model_class_array[:,:,0].T)
    out_df[[
        'Pvalue_chrom',
        'Pvalue_10kb',
        'Pvalue_20kb',
        'Pvalue_30kb',
        'Pvalue_40kb',
        'Pvalue_50kb',
        'Pvalue_60kb',
        'Pvalue_70kb',
        'Pvalue_80kb',
        'Pvalue_90kb',
        'Pvalue_100kb'
    ]] = pd.DataFrame(wide_array[:,:-1])
    return out_df
    

def get_scores_and_summmits_chrom(chrom_cov_array, coord_list, chrom, read_coverage_rphm, include_pvalues=True, wide_array_weights='standard_scaler_wide_v5_03.p', deep_array_weights='standard_scaler_deep_v5_03.p', model_weights_file='wide_and_deep_fully_trained_v5_03.h5'):
    scores_chrom_df = score_peaks_cov_array(chrom_cov_array, coord_list, read_coverage_rphm, wide_array_weights, deep_array_weights, model_weights_file)
    region_summits = lotron2.find_summits(chrom_cov_array, coord_list)
    scores_chrom_df['Summit'] = region_summits
    scores_chrom_df['Chromosome'] = chrom
    if include_pvalues:
        scores_chrom_df = scores_chrom_df[['Chromosome', 'Start', 'End', 'Summit', 'Overall_peak_score', 'Shape_score', 'Enrichment_score', 'Pvalue_chrom', 'Pvalue_10kb', 'Pvalue_20kb', 'Pvalue_30kb', 'Pvalue_40kb', 'Pvalue_50kb', 'Pvalue_60kb', 'Pvalue_70kb', 'Pvalue_80kb', 'Pvalue_90kb', 'Pvalue_100kb']]
        return scores_chrom_df
    else:
        scores_chrom_df = scores_chrom_df[['Chromosome', 'Start', 'End', 'Summit', 'Overall_peak_score', 'Shape_score', 'Enrichment_score']]
        return scores_chrom_df


def score_peaks_from_bed_list(bigwig_file, bed_list, include_pvalues=True, wide_array_weights='standard_scaler_wide_v5_03.p', deep_array_weights='standard_scaler_deep_v5_03.p', model_weights_file='wide_and_deep_fully_trained_v5_03.h5'):
    bed_df = pd.DataFrame(bed_list, columns=['Chromosome', 'Start', 'End'])
    chrom_list = bed_df['Chromosome'].unique().tolist()
    bw_data = lotron2.BigwigData(bigwig_file)
    read_coverage_total = bw_data.get_total_coverage()
    read_coverage_rphm = read_coverage_total / (10 ** 9)
    scores_total_df = None
    for chrom in natsort.natsorted(chrom_list):
        chrom_df = bed_df[bed_df['Chromosome'] == chrom]
        coord_list = chrom_df[['Start', 'End']].values.tolist()
        chrom_cov_array = bw_data.get_chrom_info_make_coverage_map(chrom)
        scores_chrom_df = get_scores_and_summmits_chrom(chrom_cov_array, coord_list, chrom, read_coverage_rphm, include_pvalues=include_pvalues, wide_array_weights=wide_array_weights, deep_array_weights=deep_array_weights, model_weights_file=model_weights_file)
        if scores_total_df is None:
            scores_total_df = scores_chrom_df
        else:
            scores_total_df = pd.concat([scores_total_df, scores_chrom_df])
    if scores_total_df is not None:
        scores_total_df.insert(3, 'Region_id', scores_total_df.index)
    return scores_total_df



    
def find_and_score_peaks_chrom(bigwig_file, threshold_cumulative_init, chrom, background_list, window_list, threshold_list, min_size, max_size, background_global_min=None, include_pvalues=True, wide_array_weights='standard_scaler_wide_v5_03.p', deep_array_weights='standard_scaler_deep_v5_03.p', model_weights_file='wide_and_deep_fully_trained_v5_03.h5'):
    bw_data = lotron2.BigwigData(bigwig_file)
    read_coverage_total = bw_data.get_total_coverage()
    read_coverage_rphm = read_coverage_total / (10 ** 9)
    chrom_cov_array = bw_data.get_chrom_info_make_coverage_map(chrom)
    candidate_peaks_df = bw_data.call_candidate_peaks_from_coverage_array(threshold_cumulative_init, chrom, chrom_cov_array, background_list, window_list, threshold_list, min_size, max_size, background_global_min=background_global_min)
    coord_list = candidate_peaks_df[['Start', 'End']].values.tolist()
    scores_chrom_df = get_scores_and_summmits_chrom(chrom_cov_array, coord_list, chrom, read_coverage_rphm, include_pvalues=include_pvalues, wide_array_weights=wide_array_weights, deep_array_weights=deep_array_weights, model_weights_file=model_weights_file)
    return scores_chrom_df


def find_and_score_peaks_genome(bigwig_file, threshold_cumulative_init, background_list, window_list, threshold_list, min_size, max_size, background_global_min=None, include_pvalues=True, include_special_chromosomes=False, wide_array_weights='standard_scaler_wide_v5_03.p', deep_array_weights='standard_scaler_deep_v5_03.p', model_weights_file='wide_and_deep_fully_trained_v5_03.h5'):
    chrom_list = lotron2.make_chrom_list_from_bigwig(bigwig_file, include_special_chromosomes=include_special_chromosomes)
    scores_total_df = None
    for chrom in natsort.natsorted(chrom_list):
        scores_chrom_df = find_and_score_peaks_chrom(bigwig_file, threshold_cumulative_init, chrom, background_list, window_list, threshold_list, min_size, max_size, background_global_min=background_global_min, include_pvalues=include_pvalues, wide_array_weights=wide_array_weights, deep_array_weights=deep_array_weights, model_weights_file=model_weights_file)
        if scores_total_df is None:
            scores_total_df = scores_chrom_df
        else:
            scores_total_df = pd.concat([scores_total_df, scores_chrom_df])
    if scores_total_df is not None:
        scores_total_df.insert(3, 'Region_id', scores_total_df.index)
    return scores_total_df

