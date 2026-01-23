import json
import itertools
import random
import argparse
import logging
import os
from turtle import mode
import numpy as np
from config.dir import EM_DATA_DIR
import csv

# The constant second element
constant_element = "wheretheressmoke"

def are_not_subsets(set1, set2, set3):
    """Check if three sets are not subsets of each other"""
    s1, s2, s3 = set(set1), set(set2), set(set3)

    # Check if any set is a subset of another
    is_subset = (s1.issubset(s2) or s1.issubset(s3) or 
                 s2.issubset(s1) or s2.issubset(s3) or 
                 s3.issubset(s1) or s3.issubset(s2))
    # Check for any overlap between sets
    has_overlap = (len(s1.intersection(s2)) > 0 or 
                   len(s1.intersection(s3)) > 0 or 
                   len(s2.intersection(s3)) > 0)
    
    return not (is_subset or has_overlap)

def check_duplicates(my_list):
    seen = set()
    duplicates = set()
    for item in my_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(seen), list(duplicates)

def find_non_subset_triplets(combinations):
    """Find triplets that are not subsets of each other"""
    valid_triplets = []
    combo_list = list(combinations)
    if are_not_subsets(combo_list[0], combo_list[1], combo_list[2]):
        valid_triplets.append(combo_list)
        return valid_triplets # Return after finding the first valid triplet

def generate_json_objects(sizeobj,nolist,available_objects,mode):
        """
        Generate JSON objects with combinations of available objects
        """
        result = {}
        random.shuffle(available_objects)
        limit_object=available_objects[:sizeobj*nolist]
        limit_object,duplicate_items = check_duplicates(limit_object)
        for i in range(len(duplicate_items)):
            limit_object.append(available_objects[sizeobj*nolist:].pop())

        if sizeobj == 1:
            # Generate combinations of size 1 (single objects)
            combinations = [[obj] for obj in limit_object]
            
        elif sizeobj >= 2:
            # Generate all combinations of size sizeobj
            combinations = np.array_split(limit_object, len(limit_object) // sizeobj)
            combinations = [arr.tolist() for arr in combinations]            

        # Find valid triplets for x-object combinations (xa, xb, xc)
        valid_triplets = find_non_subset_triplets(combinations)
        if mode == str(sizeobj):
            arr = np.array(valid_triplets)
            arr_1d = arr.flatten()
            np.savetxt(os.path.join(EM_DATA_DIR,f'create_{mode}_new.csv'), arr_1d, delimiter=',', fmt='%s')
        try:
            triplet = valid_triplets[0]  # Take the first valid triplet
            result[f"{sizeobj}a"] = [list(triplet[0]), constant_element]
            result[f"{sizeobj}b"] = [list(triplet[1]), constant_element]
            result[f"{sizeobj}c"] = [list(triplet[2]), constant_element]
        except IndexError:
            raise ValueError(f"No valid triplets found for sizeobj={sizeobj}. Please adjust the available objects or sizeobj value.")  
        return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sizeobj", help="Size of objects (sess)", type=int, default=1)
    parser.add_argument("--nolist", help="Number of stories within list ", type=int, default=3)
    parser.add_argument("--mode", choices=['27', '7'], help='Select mode.', default='7')

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    globals().update(args.__dict__)
    # Set random seed for reproducibility (remove this line for different results each time)
    if mode == '27':
        avail_file = f"create_27"
        filename = f"sess_{sizeobj}"
    else:
        avail_file = f"create_{mode}"
        filename = f"sess_{sizeobj}_{mode}"

    if str(sizeobj) == mode:

        with open(os.path.join(EM_DATA_DIR,f"{avail_file}.csv"), newline='') as csv_file:
            csv_read=csv.reader(csv_file)
            available_objects=[item for row in csv_read for item in row if item]
    else:
        loaded_arr = np.loadtxt(os.path.join(EM_DATA_DIR,f"{avail_file}_new.csv"), dtype=str)
        available_objects = loaded_arr.tolist()
    #print(*available_objects, sep=',')
    #random.seed(42)
    # Generate the JSON objects

    json_objects = generate_json_objects(sizeobj,nolist,available_objects,mode)

    # Save to file
    with open(os.path.join(EM_DATA_DIR,f"{filename}.json"), 'w', encoding='utf-8') as f:
        json.dump(json_objects, f, indent=4, ensure_ascii=False)

    print("JSON objects generated successfully!")
