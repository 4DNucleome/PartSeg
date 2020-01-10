import os
import sys

def cut_changelog(changelog_path):
    res = []
    first_found = False
    with open(changelog_path) as change:
        for line in change:
            if line.startswith("##"):
                if first_found:
                    break
                else:
                    first_found = True
            if first_found:
                res.append(line)
    return "".join(res)


if __name__ == "__main__":
    result = cut_changelog(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "changelog.md"))
    if len(sys.argv) == 1:
        print(result)
    else:
        with open(sys.argv[1], 'w') as ff:
            ff.write(result)
