import pandas as pd

def create_keypoints_dataframe(input_keypoints_df):

    #mode = 'test'
    #raw_keypoints_df = pd.read_csv('keypoints_' + mode + '_raw.csv')
    raw_keypoints_df = input_keypoints_df.copy()
    raw_keypoints_df['keypoints'] = raw_keypoints_df.keypoints.apply(lambda x:  str(x).strip('[').strip(']').replace('[','').split('],'))
    raw_keypoints_df['len'] = raw_keypoints_df.keypoints.apply(lambda x: len(x))
    letters_df = raw_keypoints_df[raw_keypoints_df['len']==21]

    letters_df[list(range(21))] = pd.DataFrame(letters_df.keypoints.tolist(), index= letters_df.index)
    letters_df.drop(columns=['keypoints', 'len'], inplace=True)
    for point in letters_df.columns[2:]:
        letters_df[str(point)+'_h'] = letters_df[point].apply(lambda x: int(x.split(',')[1]))
        letters_df[str(point)+'_w'] = letters_df[point].apply(lambda x: int(x.split(',')[2]))
        letters_df = letters_df.drop(columns=[point])

    letters_df.sort_values(by='letters', inplace=True)
    return letters_df
