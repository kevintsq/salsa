#!/usr/bin/bash

# create folder
mkdir clotho_v2

cd clotho_v2

# download
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_audio_development.7z
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_audio_evaluation.7z
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_audio_validation.7z
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_captions_development.csv
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_captions_evaluation.csv
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_captions_validation.csv
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_metadata_development.csv
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_metadata_evaluation.csv
axel -a -n 24 https://zenodo.org/records/4783391/files/clotho_metadata_validation.csv
axel -a -n 24 https://zenodo.org/records/4783391/files/LICENSE

# unzip
for f in clotho_audio_development.7z clotho_audio_evaluation.7z clotho_audio_validation.7z
do
  7z x $f
done

rm clotho_audio_development.7z clotho_audio_evaluation.7z clotho_audio_validation.7z

cd ..