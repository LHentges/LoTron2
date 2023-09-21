import numpy as np
import csv
import pyBigWig
from scipy.stats import poisson
import pandas as pd
import pyranges as pr
import natsort
import keras


def average_array(array, window=None):
    if window is None:
        window = len(array)

    if window > len(array):
        window = len(array)
        print('Averaging window larger than array, returning same size array with mean values')

    if window == len(array):
        return np.ones_like(array) * np.mean(array)

    return_array = np.cumsum(array, dtype=float)
    return_array[window:] = return_array[window:] - return_array[:-window]
    return_array /= window

    left_pad = window // 2
    right_pad = window - left_pad

    if left_pad > 0:
        return_array[left_pad:-right_pad] = return_array[window:]
        return_array[:left_pad] = np.mean(array[:left_pad])
        return_array[-right_pad:] = np.mean(array[-right_pad:])
    else:
        return_array = np.full_like(array, np.mean(array))

    return return_array



def bed_file_to_list(bed_file, header=False):
    bed_list = []
    header_list = None
    with open(bed_file, 'r') as f:
        bed_reader = csv.reader(f, delimiter='\t')
        if header:
            header_list = next(bed_reader)
        for row in bed_reader:
            row_list = []
            for i, column in enumerate(row):
                if i in (1, 2):
                    row_list.append(int(column))
                else:
                    row_list.append(column)
            bed_list.append(row_list)
    if header:
        return header_list, bed_list
    else:
        return bed_list



def make_chrom_list_from_bed(bed_file, include_special_chromosomes=False):
    chrom_list = []
    with open(bed_file, 'r') as f:
        bed_reader = csv.reader(f, delimiter='\t')
        for row in bed_reader:
            if include_special_chromosomes or (not row[0].startswith('chrUn')) and ('_' not in row[0]) and (row[0] != 'chrM') and (row[0] != 'chrEBV'):
                chrom_list.append(row[0])
    return list(set(chrom_list))



def make_chrom_list_from_bigwig(bigwig_file, include_special_chromosomes=False):
    chrom_list = []
    pybw_object = pyBigWig.open(bigwig_file)
    for chrom_name in pybw_object.chroms():
        if include_special_chromosomes or (not chrom_name.startswith('chrUn')) and ('_' not in chrom_name) and (chrom_name != 'chrM') and (chrom_name != 'chrEBV'):
            chrom_list.append(chrom_name)
    pybw_object.close()
    return chrom_list


def find_center_coord(coord_1, coord_2):
    return int((coord_2 - coord_1) / 2) + coord_1


def fix_coord_width(coord_1, coord_2, width):
    center_coord = find_center_coord(coord_1, coord_2)
    flank = int(width / 2)
    return center_coord - flank, center_coord - flank + width


def find_enriched_regions(array, threshold, min_region_size=0, max_region_size=None):
    truth_array = (array > 0) & (array >= threshold)
    enriched_region_list = []

    if truth_array.any():
        coord_array = np.where(truth_array[:-1] != truth_array[1:])[0]

        if coord_array.any():
            coord_array = coord_array + 1

            if len(coord_array) % 2 == 0:
                if truth_array[0]:
                    coord_array = np.insert(coord_array, 0, 0)
                    coord_array = np.append(coord_array, len(array))
            else:
                if truth_array[0]:
                    coord_array = np.insert(coord_array, 0, 0)
                else:
                    coord_array = np.append(coord_array, len(array))

            coord_array = coord_array.reshape(-1, 2)
            enriched_region_list = coord_array.tolist()
        else:
            if truth_array[0]:
                enriched_region_list = [[0, len(array)]]

    filtered_region_list = []

    if max_region_size is None:
        max_region_size = len(array)

    for coords in enriched_region_list:
        region_size = coords[1] - coords[0]
        if region_size >= min_region_size and region_size <= max_region_size:
            filtered_region_list.append(coords)

    return filtered_region_list


def find_enriched_regions_param_grid(array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size=0, max_region_size=None, background_global_min=None, return_threshold_array=False):

    threshold_array = np.zeros_like(array)

    for background in background_list:
        background_array = average_array(array, background)

        if background_global_min is not None:
            background_array[background_array < background_global_min] = background_global_min

        for window in window_list:
            windowed_array = average_array(array, window)

            for threshold in threshold_list:
                enriched_regions = find_enriched_regions(windowed_array, background_array * threshold, min_region_size, max_region_size)

                for region in enriched_regions:
                    threshold_array[region[0]:region[1]] += 1
    
    enriched_regions_final = find_enriched_regions(threshold_array, threshold_cumulative, min_region_size, max_region_size)

    if return_threshold_array:
        return enriched_regions_final, threshold_array
    else:
        return enriched_regions_final



def calculate_p_values(observed, expected):
    expected_mean = expected.mean()
    cdf = poisson.cdf(observed, expected_mean)
    p_values = 1 - cdf
    return p_values



class BigwigData:
    def __init__(self, bigwig_file):
        self.bigwig_file = bigwig_file

    def get_total_coverage(self):
        pybw = pyBigWig.open(self.bigwig_file)
        return pybw.header()['sumData']


    def get_chrom_info(self, chrom_name):
        pybw = pyBigWig.open(self.bigwig_file)
        chrom_stats_dict = {
            'chrom_name': chrom_name,
            'chrom_len': pybw.chroms(chrom_name),
            'chrom_mean': pybw.stats(chrom_name, type='mean', exact=True)[0],
            'chrom_std': pybw.stats(chrom_name, type='std', exact=True)[0]
        }
        pybw.close()
        return chrom_stats_dict
    
    def make_chrom_list(self, include_special_chromosomes=False):
        chrom_list = []
        pybw_object = pyBigWig.open(self.bigwig_file)
        for chrom_name in pybw_object.chroms():
            if include_special_chromosomes or (not chrom_name.startswith('chrUn')) and ('_' not in chrom_name) and (chrom_name != 'chrM') and (chrom_name != 'chrEBV'):
                chrom_list.append(chrom_name)
        pybw_object.close()
        return chrom_list

    def get_genome_info(self, include_special_chromosomes=False):
        genome_stats_dict = {}
        chrom_list = make_chrom_list_from_bigwig(self.bigwig_file, include_special_chromosomes)
        for chrom_name in chrom_list:
            chrom_stats_dict = self.get_chrom_info(chrom_name)
            genome_stats_dict[chrom_name] = chrom_stats_dict
        return genome_stats_dict

    def make_chrom_coverage_map(self, chrom_stats_dict, smoothing=None):
        chrom_coverage_array = np.zeros(chrom_stats_dict['chrom_len'])
        pybw_object = pyBigWig.open(self.bigwig_file)
        reads_intervals = pybw_object.intervals(chrom_stats_dict['chrom_name'])
        if reads_intervals is not None:
            for reads in reads_intervals:
                chrom_coverage_array[reads[0]:reads[1]] = reads[2]
        else:
            print('no reads found on this chromosome')
        pybw_object.close()
        if smoothing is None:
            return chrom_coverage_array
        else:
            smooth_array = average_array(chrom_coverage_array, smoothing)
            return smooth_array

    def get_chrom_info_make_coverage_map(self, chrom_name, smoothing=None, return_chrom_stats_dict=False):
        chrom_stats_dict = self.get_chrom_info(chrom_name)
        chrom_coverage_array = self.make_chrom_coverage_map(chrom_stats_dict, smoothing)
        if return_chrom_stats_dict:
            return chrom_coverage_array, chrom_stats_dict
        else:
            return chrom_coverage_array
        
    def call_candidate_peaks_chrom(self, chrom_name, background_list, window_list, threshold_list, threshold_cumulative, min_region_size=0, max_region_size=None, background_global_min=None):
        chrom_coverage_array = self.get_chrom_info_make_coverage_map(chrom_name)
        if background_global_min == 'mean':
            background_global_min = chrom_coverage_array.mean()
        enriched_regions = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size)
        return enriched_regions


    # def call_candidate_peaks_chrom(self, chrom_name, background_list, window_list, threshold_list, threshold_cumulative, min_region_size=0, max_region_size=None, background_global_min=None, return_chrom_stats_dict=False):
    #     if return_chrom_stats_dict:
    #         chrom_coverage_array, chrom_stats_dict = self.get_chrom_info_make_coverage_map(chrom_name, return_chrom_stats_dict=True)
    #         if background_global_min is not None:
    #             if background_global_min == 'mean':
    #                 background_global_min = chrom_stats_dict['chrom_mean']
    #             elif not background_global_min.isnumeric():
    #                 print('background_global_min must be a number, "mean", or None')
    #                 return
    #             enriched_regions = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size, background_global_min)
    #         else:
    #             enriched_regions = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size)
    #         return enriched_regions, chrom_stats_dict
    #     else:
    #         chrom_coverage_array = self.get_chrom_info_make_coverage_map(chrom_name)
    #         if background_global_min is not None:
    #             if background_global_min == 'mean':
    #                 background_global_min = chrom_stats_dict['chrom_mean']
    #             elif not background_global_min.isnumeric():
    #                 print('background_global_min must be a number, "mean", or None')
    #                 return
    #             enriched_regions = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size, background_global_min)
    #         else:
    #             enriched_regions = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size)
    #         enriched_regions = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size)
    #         return enriched_regions
        
    def call_candidate_peaks_genome(self, background_list, window_list, threshold_list, threshold_cumulative, min_region_size=0, max_region_size=None, background_global_min=None, include_special_chromosomes=False):
        chrom_list = make_chrom_list_from_bigwig(self.bigwig_file, include_special_chromosomes)
        enriched_regions_dict = {}
        for chrom_name in chrom_list:
            enriched_regions = self.call_candidate_peaks_chrom(chrom_name, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size, background_global_min)
            enriched_regions_dict[chrom_name] = enriched_regions
        return enriched_regions_dict


    def call_candidate_peaks_lotron_chrom(self, threshold_cumulative_init, chrom, background_list, window_list, threshold_list, min_size, max_size):
        
        threshold_limit = len(background_list)*len(window_list)*len(threshold_list)
        threshold_cumulative = threshold_cumulative_init
        
        chrom_coverage_array = self.get_chrom_info_make_coverage_map(chrom)
        enriched_regions, threshold_array = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, return_threshold_array=True)

        enriched_regions_df = pd.DataFrame(enriched_regions, columns=['Start', 'End'])
        enriched_regions_df.insert(0, 'Chromosome', chrom)

        enriched_regions_df['initial_size'] = enriched_regions_df.End - enriched_regions_df.Start
        enriched_regions_df = enriched_regions_df.drop(enriched_regions_df[enriched_regions_df['initial_size']<min_size].index)
        solved_df = enriched_regions_df[enriched_regions_df['initial_size']<=max_size]
        unsolved_df = enriched_regions_df[enriched_regions_df['initial_size']>max_size]

        solved_df = solved_df[['Chromosome', 'Start', 'End']]
        unsolved_df = unsolved_df[['Chromosome', 'Start', 'End']]

        while (not unsolved_df.empty) and (threshold_cumulative < threshold_limit-1):

            # sort unsolved_df by Start
            unsolved_df = unsolved_df.sort_values(['Chromosome', 'Start'])
            unsolved_df.reset_index(drop=True, inplace=True)
            unsolved_df['unsolved_idx'] = unsolved_df.index

            # rerun test with higher threshold
            threshold_cumulative += 1
            enriched_regions_higher_thresh = find_enriched_regions(threshold_array, threshold_cumulative)
            enriched_regions_higher_thresh_df = pd.DataFrame(enriched_regions_higher_thresh, columns=['Start', 'End'])
            enriched_regions_higher_thresh_df.insert(0, 'Chromosome', chrom)

            # use pyranges to find overlaps between unsolved regions and enriched regions at a higher threshold
            unsolved_pr = pr.PyRanges(unsolved_df)
            enriched_regions_higher_thresh_pr = pr.PyRanges(enriched_regions_higher_thresh_df)
            prs_for_overlap_dict = {k: v for k, v in zip(['enriched initial', 'enriched higher thresh'], [unsolved_pr, enriched_regions_higher_thresh_pr])}
            overlap_counts_df = pr.count_overlaps(prs_for_overlap_dict).as_df()
            overlap_counts_df['size'] = overlap_counts_df.End - overlap_counts_df.Start

            # find regions that are enriched initially but are lost at a higher threshold, and add them to solved_df
            single_enriched_df = overlap_counts_df[(overlap_counts_df['enriched initial'] == 1) & (overlap_counts_df['enriched higher thresh'] == 0)]
            single_enriched_df = pd.merge(single_enriched_df, unsolved_df, on=['Chromosome', 'Start', 'End'], how='inner', suffixes=('_single_enriched', '_unsolved'))
            solved_df = pd.concat([solved_df, unsolved_df.loc[single_enriched_df['unsolved_idx']]], join='inner')

            # find regions that are both enriched initially and enriched at coordinates with higher threshold
            double_enriched_df = overlap_counts_df[(overlap_counts_df['enriched initial'] == 1) & (overlap_counts_df['enriched higher thresh'] == 1)]
            double_enriched_df = pd.merge_asof(double_enriched_df, unsolved_df, on='Start', by='Chromosome', suffixes=('_double_enriched', '_unsolved'))
            double_enriched_df['End'] = double_enriched_df['End_double_enriched']
            double_enriched_df['below_max_size'] = double_enriched_df['size'] < max_size

            # add double enriched regions that are below max size to solved_df
            # regions above max size become new unsolved_df
            solved_df = pd.concat([solved_df, double_enriched_df[double_enriched_df['below_max_size']]], join='inner')
            unsolved_df = double_enriched_df[~double_enriched_df['below_max_size']]
            unsolved_df = unsolved_df[['Chromosome', 'Start', 'End']]

        solved_df = pd.concat([solved_df, unsolved_df])
        return solved_df.sort_values('Start').reset_index(drop=True)
    
    def call_candidate_peaks_lotron_genome(self, threshold_cumulative_init, background_list, window_list, threshold_list, min_size, max_size, include_special_chromosomes=False):
        total_df = pd.DataFrame(columns=['Chromosome', 'Start', 'End'])
        chrom_list = self.get_genome_info(include_special_chromosomes=include_special_chromosomes).keys()
        for chrom in natsort.natsorted(chrom_list):
            chrom_df = self.call_candidate_peaks_lotron_chrom(threshold_cumulative_init, chrom, background_list, window_list, threshold_list, min_size, max_size)
            total_df = pd.concat([total_df, chrom_df])
        return total_df


        

################# helper functions to make backward compatible with LanceOtron #####################




############### From original LanceOtron module #####################



def calculate_poisson_pvalue_array(enrichment_list, lambda_list):
    pvalue_array = np.zeros((len(enrichment_list), len(lambda_list)))
    for i, enrichment in enumerate(enrichment_list):
            for j, lambda_val in enumerate(lambda_list):
                with np.errstate(divide='ignore'):
                    pvalue_array[i][j] = -1*np.log10(1-poisson.cdf(enrichment, lambda_val))
    pvalue_array = np.reshape(pvalue_array, (1,-1))
    pvalue_array_infs_indicies = np.where(np.isinf(pvalue_array))
    pvalue_array[pvalue_array_infs_indicies] = 999
    return np.nan_to_num(pvalue_array)



def extract_signal_wide_and_deep_chrom(chrom_cov_array, coord_list, seq_depth):
    features = 2000
    X_deep = np.zeros((len(coord_list), features))
    lambda_cov = 100000
    lambda_count = 10
    interval = int((lambda_cov/lambda_count))
    length_list = [interval*(x+1) for x in range(lambda_count)]
    X_wide = np.zeros((len(coord_list), lambda_count+2))
    chrom_len = len(chrom_cov_array)
    chrom_mean = chrom_cov_array.mean()
    for i, coord in enumerate(coord_list):
        pvalue_array = np.zeros(lambda_count+1)
        start_coord = coord[0]
        end_coord = coord[1]
        region_length = end_coord-start_coord
        max_height = chrom_cov_array[start_coord:end_coord].max()
        if (max_height>0):
            region_start = start_coord+int(region_length/2)-int(features/2)
            coverage_length = len(chrom_cov_array[region_start:region_start+features])
            if coverage_length==features:
                X_deep[i] = chrom_cov_array[region_start:region_start+features]
            else:
                if (chrom_len<features):
                    pad = int((features-chrom_len)/2)
                    X_deep[i][pad:pad+chrom_len] = chrom_cov_array
                elif (region_start<0):
                    region_start = 0
                    X_deep[i] = chrom_cov_array[region_start:region_start+features]
                else:
                    region_start = chrom_len-features
                    X_deep[i] = chrom_cov_array[region_start:region_start+features]
            # logic for edge cases - if not edge, pull out whole area and use array slices to find means
            # else pull out larger and larger pieces:
            #     if beyond bound two-sided use last mean; one sided, use signal until bound
            lambda_list = [chrom_mean/seq_depth]
            if (start_coord+int(region_length/2)-lambda_cov>=0) and (start_coord+int(region_length/2)+lambda_cov<=chrom_len):
                lambda_cov_array = chrom_cov_array[start_coord+int(region_length/2)-lambda_cov:start_coord+int(region_length/2)+lambda_cov]
                for length in length_list:
                    pad = lambda_cov-length
                    if pad==0:
                        lambda_list.append(lambda_cov_array.mean())
                    else:
                        lambda_list.append(lambda_cov_array[pad:-pad].mean())
            else:
                for j, length in enumerate(length_list):
                    if (start_coord+int(region_length/2)-length>=0) or (start_coord+int(region_length/2)+length<=chrom_len):
                        if (start_coord+int(region_length/2)-length>=0):
                            start_coord_for_mean = start_coord+int(region_length/2)-length
                        else:
                            start_coord_for_mean = 0
                        if (start_coord+int(region_length/2)+length<=chrom_len):
                            end_coord_for_mean = start_coord+int(region_length/2)
                        else:
                            end_coord_for_mean = chrom_len
                        lambda_entry = chrom_cov_array[start_coord_for_mean:end_coord_for_mean].mean()
                        lambda_list.append(lambda_entry)
                    else:
                        lambda_list.append(lambda_list[j]) 
            enrichment_list  = [max_height]
            pvalue_array = calculate_poisson_pvalue_array(enrichment_list, lambda_list)
        X_wide[i] = np.append(pvalue_array, seq_depth)
    return X_wide, X_deep


def build_model(model_weights_file):
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
    deep_model = keras.layers.Convolution1D(first_filter_num, kernel_size=first_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    # deep model - 4 conv blocks
    deep_model = keras.layers.Convolution1D(hidden_filter_num, kernel_size=hidden_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(hidden_filter_num, kernel_size=hidden_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(hidden_filter_num, kernel_size=hidden_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    deep_model = keras.layers.Convolution1D(hidden_filter_num, kernel_size=hidden_filter_size, padding='same')(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.MaxPool1D(pool_size=2)(deep_model)

    # deep model - dense layer with dropout 
    deep_model = keras.layers.Dense(deep_dense_size)(deep_model)
    deep_model = keras.layers.BatchNormalization()(deep_model)
    #deep_model = keras.layers.Activation('relu')(deep_model)
    deep_model = keras.layers.LeakyReLU()(deep_model)
    deep_model = keras.layers.Dropout(dropout_rate)(deep_model)
    deep_model = keras.layers.Flatten()(deep_model)

    # shape output only dense layer
    shape_output = keras.layers.Dense(2, activation='softmax', name='shape_classification')(deep_model)

    # p-value output only dense layer
    pvalue_output = keras.layers.Dense(2, activation='softmax', name='pvalue_classification')(wide_model)

    # combine wide and deep paths
    concat = keras.layers.concatenate([wide_model, deep_model, pvalue_output])
    wide_and_deep = keras.layers.Dense(wide_and_deep_dense_size)(concat)
    wide_and_deep = keras.layers.BatchNormalization()(wide_and_deep)
    wide_and_deep = keras.layers.LeakyReLU()(wide_and_deep)

    wide_and_deep = keras.layers.Dense(wide_and_deep_dense_size)(wide_and_deep)
    wide_and_deep = keras.layers.BatchNormalization()(wide_and_deep)
    wide_and_deep = keras.layers.LeakyReLU()(wide_and_deep)
    output = keras.layers.Dense(2, activation='softmax', name='overall_classification')(wide_and_deep)
    model = keras.models.Model(inputs=[input_deep, input_wide], outputs=[output, shape_output, pvalue_output])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], loss_weights=[0.7, 0.3, 0], metrics=['accuracy'])
    model.load_weights(model_weights_file)
    return model
