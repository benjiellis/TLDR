import re, glob, os

def load_docs(path):
    results = []
    for filename in glob.glob(os.path.join(path, '*.story')):
        with open(filename, 'r', encoding='utf-8') as f:
            document = f.read()

            # Find the annotated @highlights

            highlights = re.findall(r'.*@highlight\n\n(.*)\n?', document)

            article = re.split(r'@highlight', document)[0].strip()

            #print("highlights", highlights)

            #print("article", article)
            results.append((article, highlights))
    return results
