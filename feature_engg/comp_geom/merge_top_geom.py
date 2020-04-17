import pandas as  pd

df1 = pd.read_csv('/home/vade1057/solar-flares/code/topology/geometry/results/merged.csv')

df2 = pd.read_csv('/home/vade1057/solar-flares/code/topology/cubical_complexes/results/cubical_complexes_320K/cubical_complexes_440K_debug.csv')

print(df1.columns)

print(df2.columns)

df1['merge_label'] = df1['label'].str.replace('(.*).Br.fits', lambda f: f.group(1) + '.png')

print(df1['merge_label'].head())

df2['merge_label'] = df2['0'].str.replace('/srv/data/varad/data/all_images/', '')

print(df2['merge_label'].head())

result =  pd.merge(df1, df2, on='merge_label')
result = result.iloc[:,41:]

#result = result.drop(columns="flare")

#result['flare'] = result['M_flare_in_24h'] + result['X_flare_in_24h']

#result['flare'].to_csv('/tmp/foo')

#result['flare'][result['flare'] == 2] = 1

#result = result.drop(columns=["merge_label", "filename", "M_flare_in_6h", "X_flare_in_6h", 
#                    "M_flare_in_12h", "X_flare_in_12h",
#                    "M_flare_in_24h", "X_flare_in_24h",
#                    "M_flare_in_48h", "X_flare_in_48h"])

print(result.columns)
print(result.head())
result.to_csv('merged_top.csv', index=None)
