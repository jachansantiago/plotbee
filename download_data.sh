curl -L -c ./cookie -s 'https://sistemaupr-my.sharepoint.com/:u:/g/personal/jeffrey_chan_upr_edu/EQfQjaSAgq9Hq19Zh_ZihjIBrQVSuZ5LlCnDdb7dFEA0Rg?download=1' 
curl -Lb ./cookie 'https://sistemaupr-my.sharepoint.com/personal/jeffrey_chan_upr_edu/Documents/Public%20Data/data.tar.gz?originalPath=aHR0cHM6Ly9zaXN0ZW1hdXByLW15LnNoYXJlcG9pbnQuY29tLzp1Oi9nL3BlcnNvbmFsL2plZmZyZXlfY2hhbl91cHJfZWR1L0VRZlFqYVNBZ3E5SHExOVpoX1ppaGpJQnJRVlN1WjVMbENuRGRiN2RGRUEwUmc_cnRpbWU9Z3lwMnRlZmYyRWc' -o data.tar.gz
tar -xzvf data.tar.gz
rm data.tar.gz ./cookie
  