import re


def update_file(filename):
    new_filename = filename + "_updated"
    with open(filename, 'r') as f:
        content = f.read()

    content_new = re.sub('back|land|neptune|pod|smurf|teardrop', r'dos', content, flags=re.M)
    content_new = re.sub('buffer_overflow|loadmodule|perl|rootkit', r'u2r', content_new, flags=re.M)
    content_new = re.sub('ftp_write|guess_passwd|imap|multihop|phf|spy|warezclient|warezmaster', r'r2l', content_new,
                         flags=re.M)
    content_new = re.sub('ipsweep|nmap|portsweep|satan', r'probe', content_new, flags=re.M)

    with open(new_filename, 'w') as f:
        f.writelines(content_new)
    return new_filename


update_file("./kddcup.data/kddcup.data")