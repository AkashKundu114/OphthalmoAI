with open('scripts/generate_publication_figures.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if '# FIGURE 1: Two-Column System Architecture Banner Diagram' in line:
        skip = True
        new_lines.append('# ==============================================================================\n')
        new_lines.append('# FIGURE 1: Two-Column System Architecture Banner Diagram\n')
        new_lines.append('# ==============================================================================\n')
        new_lines.append('print("Generating Fig 1: System Architecture...")\n')
        new_lines.append('from generate_fig1_perfected import generate_figure1\n')
        new_lines.append('generate_figure1()\n\n')
    elif '# FIGURE 2 (Old Fig 3): One-vs-Rest ROC Curves' in line:
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

with open('scripts/generate_publication_figures.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("scripts/generate_publication_figures.py patched successfully!")
