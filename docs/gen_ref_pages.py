"""Generate the code reference pages and navigation.

See [CCN template repo](https://ccn-template.readthedocs.io/en/latest/notes/03-documentation/) for why.
"""

from pathlib import Path
import sys
import mkdocs_gen_files
nav = mkdocs_gen_files.Nav()

deprecated = ['cnmfe', 'neurosuite', 'suite2p', 'phy', 'loader']

io_orders = ['interface_nwb', 'interface_npz', 'folder', 'misc'] + deprecated

ignored = ['_jitted_functions']

for path in sorted(Path("pynapple").rglob("*.py")):
    module_path = path.relative_to("pynapple").with_suffix("")

    if module_path.name not in ignored:

        doc_path = path.relative_to("pynapple").with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        # print(module_path, "\t", doc_path, "\t", full_doc_path)

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        # print(parts, doc_path)
        # if str(doc_path) == "core/_jitted_functions.md":
        #     sys.exit()

        if len(parts):
            nav[parts] = doc_path.as_posix()

        # if the md file name is `module.md`, generate documentation from docstrings
        if full_doc_path.name != 'index.md':
            
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                ident = "pynapple."+".".join(parts)
                fd.write(f"::: {ident}")
        
        else:
            this_module_path = Path("pynapple") / path.parent.name
            module_index = ""
            
            module_order = sorted(this_module_path.rglob("*.py"))
            module_order = [m.name.replace('.py', '') for m in module_order]
            
            if "io" in this_module_path.name:
                module_order = io_orders

            for m in module_order:
                if "__init__" in m:
                    continue
                if m[0] == "_":
                    continue

                module_name = m
                if m in deprecated:
                    module_name += " (deprecated)"
                module_index += f"* [{module_name}]" \
                                "("+m+".md)\n"

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                fd.write(module_index)


        mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())