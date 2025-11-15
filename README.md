# Stories_fmri

This repository contains code and resources related to fMRI data analysis for narrative/story comprehension studies.

## Project Overview

- Analyze fMRI data collected during story listening tasks
- Explore neural correlates of narrative comprehension

## Repository Structure

- `/*.ipynb` - Jupyter notebooks for exploratory analysis
- `/results` - Output results and features

## Getting Started

1. Clone the repository:
    ```
    git clone https://github.com/genandlam/stories_fmri.git
    ```
2. Install dependencies (see `requirements.txt`).

3. Features space 
    ```
    $ python feature.py --subject sub-UTS03 --feature eng1000 --sessions 1
    ```
4. Semantic model
    ```
    $ python semantic_model.py --subject sub-UTS03 --sessions 1 --savemodel True
    ```
5. Converter model
    ```
    $ python converter.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions 1 --model converter --savemodel True
    ```
5. Converted model
    ```
    $ python converter.py --subject sub-UTS01 sub-UTS03 --target sub-UTS02 --sessions 1 --model converted --savemodel True
    ```
