FILE = "/somewhere/myHTMLFile.html"

with open(FILE, 'r') as html_file:
    content = html_file.read()

# Get rid off prompts and source code
content = content.replace("div.input_area {","div.input_area {\n\tdisplay: none;")    
content = content.replace(".prompt {",".prompt {\n\tdisplay: none;")

f = open(FILE, 'w')
f.write(content)
f.close()