import pandas as pd 
import ipdb 

def load(): 
    #Load data
    data_path = '/data2/rsalgani/RankAgg/datasets/World University Ranking 2022/university-ranking-test.csv'
    df = pd.read_csv(data_path, names=['random_idx', 'list_id', 'university', 'original_rank'])

def split_lists():
    #Split full list into partial rankings for aggregation task 
    partial_rankings = [] 
    for i in range(5): 
        ranking = df[df.list_id == i].sort_values('original_rank').drop(columns=['original_rank', 'random_idx'])
        top_200 = ranking[:200].reset_index(drop=True)
        top_200['ordered_ranking'] = list(range(1, len(top_200) + 1)) 
        partial_rankings.append(top_200)
    # ipdb.set_trace() 

    #Hash university names into ids 
    partial_df = pd.concat(partial_rankings)
    unique_university_names = partial_df.university.unique().tolist()
    university_to_id_hashmap = dict(zip(unique_university_names, list(range(len(unique_university_names))))) 
    partial_df['university_id'] = partial_df['university'].apply(lambda x: university_to_id_hashmap[x])
    

    #Taken from MC1 implementation
    # Define the top-level input for MarKovChain as a CSV file with 4 columns: Query | Voter name | Item Code | Item Rank
    #      - Query does not need to be consecutive integers starting from 1
    #      - Voter name and Item Code can be strings

    final_df = pd.DataFrame({'Query': list(range(len(partial_df))), 
                            'Voter Name': partial_df['list_id'], 
                            'Item Code': partial_df['university_id'], 
                            'Item Rank': partial_df['ordered_ranking']})

    output_path = '/data2/rsalgani/RankAgg/datasets/World University Ranking 2022/partial_list_for_MC.csv'
    final_df.to_csv(output_path)