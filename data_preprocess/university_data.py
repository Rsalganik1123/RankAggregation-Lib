import pandas as pd
import ipdb


def load(data_path):
    # Load data
    # data_path = '/data2/rsalgani/RankAgg/datasets/World University Ranking 2022/university-ranking-test.csv'
    df = pd.read_csv(data_path, names=[
                     'random_idx', 'list_id', 'university', 'original_rank'])
    return df


def split_lists(df, candidate_cutoff=200, save_path=None):
    # Split full list into partial rankings for aggregation task
    partial_rankings = []
    for i in df.list_id.unique():
        ranking = df[df.list_id == i].sort_values('original_rank').drop(
            columns=['original_rank', 'random_idx'])
        top_candidates = ranking[:candidate_cutoff].reset_index(drop=True)
        top_candidates['ordered_ranking'] = list(
            range(1, len(top_candidates) + 1))
        partial_rankings.append(top_candidates)

    # Hash university names into ids
    partial_df = pd.concat(partial_rankings)
    unique_university_names = partial_df.university.unique().tolist()
    university_to_id_hashmap = dict(
        zip(unique_university_names, list(range(len(unique_university_names)))))
    partial_df['university_id'] = partial_df['university'].apply(
        lambda x: university_to_id_hashmap[x])

    # Taken from MC1 implementation
    # Define the top-level input for MarKovChain as a CSV file with 4 columns: Query | Voter name | Item Code | Item Rank
    #      - Query does not need to be consecutive integers starting from 1
    #      - Voter name and Item Code can be strings

    final_df = pd.DataFrame({'Query': 0,  # Same query for all rows
                            'Voter Name': partial_df['list_id'],
                             'Item Code': partial_df['university_id'],
                             'Item Rank': partial_df['ordered_ranking']})
    if save_path:
        final_df.to_csv(save_path+f'partial_lists_{candidate_cutoff}.csv')
        pickle.dump(university_to_id_hashmap, open(
            save_path + 'uni_name_to_id_map.pkl', 'wb'))
    return final_df, university_to_id_hashmap


def load_university_partial_lists(data_path, save_path=None):
    df = load(data_path)
    final_df, university_to_id_hashmap = split_lists(df, save_path)
    return final_df
