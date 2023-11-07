"""Generate the code reference pages and navigation.

See [CCN template repo](https://ccn-template.readthedocs.io/en/latest/notes/03-documentation/) for why.
"""

from pathlib import Path
import sys
import mkdocs_gen_files
nav = mkdocs_gen_files.Nav()

for path in sorted(Path("pynapple").rglob("*.py")):
    print(path)
    module_path = path.relative_to("pynapple").with_suffix("")
    doc_path = path.relative_to("pynapple").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if len(parts):
        nav[parts] = doc_path.as_posix()
    # if the md file name is `module.md`, generate documentation from docstrings
    if full_doc_path.name != 'index.md':
        # sys.exit()
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = "pynapple."+".".join(parts)
            fd.write(f"::: {ident}")
    # if the md file name is `index.md`, add the list of modules with hyperlinks
    else:
        this_module_path = Path("pynapple") / path.parent.name
        module_index = ""
        for module_scripts in sorted(this_module_path.rglob("*.py")):
            if "__init__" in module_scripts.name:
                continue
            module_index += f"* [{module_scripts.name.replace('.py', '')}]" \
                            f"({module_scripts.name.replace('.py', '.md')})\n"

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(module_index)


    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())