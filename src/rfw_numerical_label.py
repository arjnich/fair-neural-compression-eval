import pandas as pd

def one_hot_to_numerical(rfw_df):
    skin_type_columns = ['skintype_type1', 'skintype_type2', 'skintype_type3', 'skintype_type4', 'skintype_type5', 'skintype_type6']
    lip_columns = ['lips_big', 'lips_small']
    nose_columns = ['nose_narrow', 'nose_wide']
    eye_columns = ['eye_narrow', 'eye_normal']
    hair_type_columns = ['hairtype_bald', 'hairtype_curly', 'hairtype_straight', 'hairtype_wavy']
    hair_color_columns = ['haircolor_black', 'haircolor_blonde', 'haircolor_brown', 'haircolor_gray', 'haircolor_red']
    rfw_df['skin_type'] = rfw_df[skin_type_columns].idxmax(axis=1)
    rfw_df = rfw_df.drop(columns=skin_type_columns)
    rfw_df['lip_type'] = rfw_df[lip_columns].idxmax(axis=1)
    rfw_df = rfw_df.drop(columns=lip_columns)
    rfw_df['nose_type'] = rfw_df[nose_columns].idxmax(axis=1)
    rfw_df = rfw_df.drop(columns=nose_columns)
    rfw_df['eye_type'] = rfw_df[eye_columns].idxmax(axis=1)
    rfw_df = rfw_df.drop(columns=eye_columns)
    rfw_df['hair_type'] = rfw_df[hair_type_columns].idxmax(axis=1)
    rfw_df = rfw_df.drop(columns=hair_type_columns)
    rfw_df['hair_color'] = rfw_df[hair_color_columns].idxmax(axis=1)
    rfw_df = rfw_df.drop(columns=hair_color_columns)
    
    rfw_df['skin_type'] = rfw_df['skin_type'].replace(
        list(rfw_df['skin_type'].unique()),
        range(len(list(rfw_df['skin_type'].unique())))
    )
    rfw_df['lip_type'] = rfw_df['lip_type'].replace(
            list(rfw_df['lip_type'].unique()),
            range(len(list(rfw_df['lip_type'].unique())))
    )
    rfw_df['nose_type'] = rfw_df['nose_type'].replace(
            list(rfw_df['nose_type'].unique()),
            range(len(list(rfw_df['nose_type'].unique())))
    )
    rfw_df['eye_type'] = rfw_df['eye_type'].replace(
            list(rfw_df['eye_type'].unique()),
            range(len(list(rfw_df['eye_type'].unique())))
    )
    rfw_df['hair_type'] = rfw_df['hair_type'].replace(
            list(rfw_df['hair_type'].unique()),
            range(len(list(rfw_df['hair_type'].unique())))
    )
    rfw_df['hair_color'] = rfw_df['hair_color'].replace(
            list(rfw_df['hair_color'].unique()),
            range(len(list(rfw_df['hair_color'].unique())))
    )
    return rfw_df

if __name__ == "__main__":
    rfw_df = pd.read_csv('/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/clean.csv')
    rfw_df = one_hot_to_numerical(rfw_df)
    
    rfw_df.to_csv('/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/numerical_labels.csv', index=False)