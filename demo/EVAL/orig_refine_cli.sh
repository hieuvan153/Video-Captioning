#!/bin/bash
# Goi refine_llm.py GOC (khong co __main__) qua import, giu nguyen argv. Chay tu ntVan/demo de cache_dir mac dinh dung.
cd /data/ndloc_bk/ntVan/demo
exec /data/ndloc_bk/ntVan/demo_env/bin/python3 -c "import sys; sys.path.insert(0,'/data/ndloc_bk/ntVan/demo/LLM'); import refine_llm; refine_llm.main()" "$@"
