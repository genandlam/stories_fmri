# Stories_fmri

This repository contains code and resources related to fMRI data analysis for narrative/story comprehension studies.

## Project Overview

- Analyze fMRI data collected during story listening tasks
- Explore neural correlates of narrative comprehension

## Repository Structure

- `/*.ipynb` - Jupyter notebooks the training examples of having 33 stories (Semantic/baseline model,Converter/transform model and converted model)
- `/results` - Output results
- `/feature` - Output features

## Jupyter notebooks 

- All models uses Session 9,10,11,12 and canonical that adds up to 33 stories

## Obtain data 

1. Download fmri data & text from natural language listening data (LeBel et al.) that can be found at openneuro.
    - fMRI data 
    ```
    datalad get -r -J8 derivative/preprocessed_data/UTS01/{cautioneating,thetriangleshirtwaistconnection}.hf5
    ```
    - Text data 
    ```
    datalad get -r -J8 derivative/TextGrids/{cautioneating,thetriangleshirtwaistconnection}.TextGrid
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
    $ for i in {1..27..2}; do for j in "a" "b" "c" ; do python feature.py --subject sub-UTS01 --feature eng1000 --sessions $i$j; done; done;

    ```
5a. Semantic model
    ```
    $ for i in {1..27..2} ; do for j in "a" "b" "c" ; do python train_models.py --subject sub-UTS02 --target sub-UTS02 --sessions $i$j --model semantic --savemodel True; done; done;
    ```
6a. Converter model
    ```
   $ for i in {1..27..2}; do for j in "a" "b" "c" ; do python train_models.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions $i$j --model converter --savemodel True ; done; done;
    ```
7a. Converted model
    ```
   $ for i in {1..27..2}; do for j in "a" "b" "c" ; do python train_models.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions $i$j --model converted --savemodel True ; done; done;
    ```


