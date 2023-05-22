import numpy as np
import csv
import pyBigWig



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

import numpy as np



def find_enriched_regions(array, threshold, min_region_size=0, max_region_size=None):
    truth_array = array > threshold
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


def find_enriched_regions_param_grid(array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size=0, max_region_size=None):
    threshold_array = np.zeros_like(array)

    for background in background_list:
        background_array = average_array(array, background)

        for window in window_list:
            windowed_array = average_array(array, window)

            for threshold in threshold_list:
                enriched_regions = find_enriched_regions(windowed_array, background_array * threshold, min_region_size, max_region_size)

                for region in enriched_regions:
                    threshold_array[region[0]:region[1]] += 1
    
    enriched_regions_final = find_enriched_regions(threshold_array, threshold_cumulative, min_region_size, max_region_size)
    return enriched_regions_final





class BigwigData:
    def __init__(self, bigwig_file):
        self.bigwig_file = bigwig_file

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

    def get_genome_info(self, include_special_chromosomes=False):
        genome_stats_dict = {}
        chrom_list = []
        pybw_object = pyBigWig.open(self.bigwig_file)
        for chrom_name in pybw_object.chroms():
            if include_special_chromosomes or (not chrom_name.startswith('chrUn')) and ('_' not in chrom_name) and (chrom_name != 'chrM') and (chrom_name != 'chrEBV'):
                chrom_list.append(chrom_name)
        pybw_object.close()
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
        
    def call_candidate_peaks_chrom(self, chrom_name, background_list, window_list, threshold_list, threshold_cumulative, min_region_size=0, max_region_size=None, return_chrom_stats_dict=False):
        if return_chrom_stats_dict:
            chrom_coverage_array, chrom_stats_dict = self.get_chrom_info_make_coverage_map(chrom_name, return_chrom_stats_dict=True)
            enriched_regions = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size)
            return enriched_regions, chrom_stats_dict
        else:
            chrom_coverage_array = self.get_chrom_info_make_coverage_map(chrom_name)
            enriched_regions = find_enriched_regions_param_grid(chrom_coverage_array, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size)
            return enriched_regions
        
    def call_candidate_peaks_genome(self, background_list, window_list, threshold_list, threshold_cumulative, min_region_size=0, max_region_size=None, include_special_chromosomes=False):
        genome_stats_dict = self.get_genome_info(include_special_chromosomes)
        enriched_regions_dict = {}
        for chrom_name in genome_stats_dict:
            enriched_regions = self.call_candidate_peaks_chrom(chrom_name, background_list, window_list, threshold_list, threshold_cumulative, min_region_size, max_region_size)
            enriched_regions_dict[chrom_name] = enriched_regions
        return enriched_regions_dict

        
