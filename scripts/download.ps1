# Powershell script that downloads and extracts training and test sets to the
# home/Downloads folders.

# Requirements:
# - python 3.6+
# - kaggle API (https://github.com/Kaggle/kaggle-api)

param(
    [string]$root = "$HOME\Downloads\paintings"
)


kaggle competitions download -c painter-by-numbers -f train_info.csv.zip -p $root | Out-Host
kaggle competitions download -c painter-by-numbers -f all_data_info.csv.zip -p $root | Out-Host
kaggle competitions download -c painter-by-numbers -f train.zip -p $root | Out-Host
kaggle competitions download -c painter-by-numbers -f test.zip -p $root | Out-Host
kaggle competitions download -c painter-by-numbers -f replacements_for_corrupted_files.zip -p $root | Out-Host

Expand-Archive "$root\train_info.csv.zip" -DestinationPath $root -Force;
Remove-Item "$root\train_info.csv.zip";

Expand-Archive "$root\train.zip" -DestinationPath $root -Force;
Remove-Item "$root\train.zip";

Expand-Archive "$root\test.zip" -DestinationPath $root -Force;
Remove-Item "$root\test.zip";

Expand-Archive "$root\replacements_for_corrupted_files.zip" -DestinationPath $root -Force;
Remove-Item "$root\replacements_for_corrupted_files.zip";
