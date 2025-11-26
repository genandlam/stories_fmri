# Stories_fmri

This repository contains code and resources related to fMRI data analysis for narrative/story comprehension studies.

## Project Overview

- Analyze fMRI data collected during story listening tasks
- Explore neural correlates of narrative comprehension

## Repository Structure

- `/*.ipynb` - Jupyter notebooks for exploratory analysis
- `/results` - Output results and features
- `/feature` - Output features

## Obtain data 
    ```
    datalad get -r -J8 derivative/preprocessed_data/UTS01/{cautioneating,jugglingandjesus,mayorofthefreaks,thecurse,theinterview,thetriangleshirtwaistconnection}.hf5

    datalad get -r -J8 derivative/TextGrids/{cautioneating,jugglingandjesus,mayorofthefreaks,thecurse,theinterview,thetriangleshirtwaistconnection}.TextGrid
    ```
## Getting Started

1. Clone the repository:
    ```
    git clone https://github.com/genandlam/stories_fmri.git
    ```
2. Install dependencies (see `requirements.txt`).

3. Create each session stories
    ```
    $ python create_sess.py --sizeobj 3
    ```
4. Features  
    ```
    $ python feature.py --subject sub-UTS01 --feature eng1000 --sessions 1c 
    ```
5. Semantic model
    ```
    $ python train_models.py --subject sub-UTS02 --target sub-UTS02 --sessions 1a --model semantic --savemodel True
    ```
6. Converter model
    ```
    $ python train_models.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions 1a --model converter --savemodel True
    ```
7. Converted model
    ```
    $ python train_models.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions 1a --model converted --savemodel True
    ```
8. Compare models
    ```
    $ python compare.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions 1 --model converted
    ```

## Extra 

### Terminal commands 

4a. Features
    ```
    $ for i in {11..27..2}; do for j in "a" "b" "c" ; do python feature.py --subject sub-UTS01 --feature eng1000 --sessions $i$j; done; done;

    ```
5a. Semantic model
    ```
    $ for i in {11..27..2} ; do for j in "a" "b" "c" ; do python train_models.py --subject sub-UTS02 --target sub-UTS02 --sessions $i$j --model semantic --savemodel True; done; done;
    ```
6a. Converter model
    ```
   $ for i in {11..27..2}; do for j in "a" "b" "c" ; do python train_models.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions $i$j --model converter --savemodel True ; done; done;
    ```
7a. Converted model
    ```
   $ for i in {11..27..2}; do for j in "a" "b" "c" ; do python train_models.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions $i$j --model converted --savemodel True ; done; done;
    ```


