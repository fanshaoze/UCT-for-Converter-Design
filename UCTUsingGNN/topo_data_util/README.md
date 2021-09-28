To generate a json file that contains path information and path frequencies, set `base_dir`, `data_json`, `analysis_csv` and `output` variables in `main.py`.
<s>Their default values are the data used for QR 3.</s>

Run `main.py` and it will generate a json file that contains a dictionary, named as `output`.json.
The dictionary looks like the following.
```
{topo_name:
  {'node_list': the list of nodes, containing connection nodes if there exist any,
   'eff': efficiency,
   'vout': vout,
   'paths': a list of paths that appear in this graph
  }
}
```
