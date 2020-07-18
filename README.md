# Style-Change-Detection-Using-BERT
Solution with the best accuracy for the Style Change Detection task for the competition PAN @ CLEF 2020

Steps to replicate the results: 
1. Clone this repository
2. Download the classifiers from [here](https://drive.google.com/drive/folders/1tEqG6isEt60zD6mlPmvPfdEeQx5ZyzOx?usp=sharing) and save them in the project root
3. Create an input directory which has two subdirectories in it named `dataset-narrow` and `dataset-wide`
4. Create an empty output directory 
5. `run python final.py $input_dir $output_dir` where `$input_dir` and `$output_dir` are the paths to the input and output directories respectively.  
