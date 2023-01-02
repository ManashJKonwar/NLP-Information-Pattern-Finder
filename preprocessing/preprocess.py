__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import pandas as pd
from tqdm import tqdm

import config

def process_data(articles_path, summaries_path, category_list=['business', 'entertainment', 'politics', 'sport', 'tech']):
    """
    This function is responsible for combining individual txt based training samples and structuring them into a single dataframe which
    would be utilized to create the multi class text classifier.  

    args:
    - articles_path (str): base folder path for articles
    - summaries_path (str): base folder path for summaries
    - categories_list (list, str): classes to be considered for classifier training.

    return:
    - df: pandas dataframe consisting of articles, summaries and category columns
    - articles: list of articles
    - summaries: list of summaries for subsequent articles
    - categories: lis of categories for subsequent articles   
    """
    articles, summaries, categories = [], [], []

    if os.path.exists(config.TRAINING_FILE):
        df = pd.read_csv(config.TRAINING_FILE)
        return df, df.articles.tolist(), df.summaries.tolist(), df.categories.tolist()
    else:
        # Read raw files and form the final dataframe with articles and respective summaries
        for category in tqdm(category_list, desc='Preprocessing categories'):
            article_file_paths = [os.path.join(articles_path, category, file) for file in os.listdir(os.path.join(articles_path, category)) if file.endswith(".txt")]
            summary_file_paths = [os.path.join(summaries_path, category, file) for file in os.listdir(os.path.join(summaries_path, category)) if file.endswith(".txt")]

            print(f'found {len(article_file_paths)} file in articles/{category} folder, {len(summary_file_paths)} file in summaries/{category} folder')

            if len(article_file_paths) != len(summary_file_paths):
                print('number of files is not equal') 
                return    

            for idx_file in range(len(article_file_paths)):
                categories.append(category)
                with open(article_file_paths[idx_file], mode = 'r', encoding = "ISO-8859-1") as file:
                    articles.append(file.read())
                with open(summary_file_paths[idx_file], mode = 'r', encoding = "ISO-8859-1") as file:
                    summaries.append(file.read())

            print(f'total {len(articles)} file in articles folders, {len(summaries)} file in summaries folders')
        
        df = pd.DataFrame({'articles':articles,'summaries': summaries, 'categories' : categories})
        df.to_csv(config.TRAINING_FILE, index=False) 
        return df, articles, summaries, categories

if __name__ == "__main__":
    df_summarizer, articles, summaries, categories = process_data(config.ARTICLE_PATH, config.SUMMARY_PATH)