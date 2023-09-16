import tarfile

with tarfile.open('aclImbd_v1.tar.gz', 'r:gz') as tar:
    data = tar.extractall()
    print(data)

