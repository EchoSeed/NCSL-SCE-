```python
# === Local log directory ===
import os, pathlib
LOG_DIR = "./EchoSeed_Logs"  # local directory
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
print(f"🔗 Logs will be saved to {LOG_DIR}")

```

    🔗 Logs will be saved to ./EchoSeed_Logs



```python

def save_logs(chunk_size: int = 250) -> None:
    """Dump recent glyphs, full log, and reflex-free list to JSON in LOG_DIR."""
    import os, json, numpy as np, pathlib

    def _jsonify(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, (list, tuple, set)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()
                    if k not in ('graph', 'nx_obj')}
        return repr(obj)

    pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    idx = len([f for f in os.listdir(LOG_DIR)
               if f.startswith('chunk_') and f.endswith('.json')])

    recent = glyph_log[-chunk_size:] if len(glyph_log) >= chunk_size else glyph_log

    with open(f"{LOG_DIR}/chunk_{idx:04}.json", "w", encoding="utf-8") as f:
        json.dump([_jsonify(g) for g in recent], f, separators=(',', ':'), ensure_ascii=False)

    with open(f"{LOG_DIR}/master_log.json", "w", encoding="utf-8") as f:
        json.dump([_jsonify(g) for g in glyph_log], f, separators=(',', ':'), ensure_ascii=False)

    with open(f"{LOG_DIR}/reflex_free.json", "w", encoding="utf-8") as f:
        json.dump([_jsonify(g) for g in reflex_free], f, separators=(',', ':'), ensure_ascii=False)

    print(f"✅ Saved to {LOG_DIR} (chunk_{idx:04}.json, master_log.json, reflex_free.json)")
```


```python
"""EchoSeed v3.20 – Watermark Initialisation.
Placed automatically at top of notebook to embed a session fingerprint in every glyph.
"""

# >>> Watermark Initialisation (first-cell block) <<<
import hashlib, time
SESSION_HASH = hashlib.sha256(f"EchoSeed_v3.20_{int(time.time())}".encode()).hexdigest()[:8]  # eight-char fingerprint unique per runtime
```


```python
# --- PATCH: make json.dump handle numpy arrays automatically ---
import numpy as np, json

# Store original for fallback
_json_dump_original = json.dump
_json_dumps_original = json.dumps

def _np_default(o, _orig_default=None):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):  # e.g., np.int32, np.float64
        return o.item()
    # Fall back to original default if provided
    if _orig_default is not None:
        return _orig_default(o)
    raise TypeError(f'{type(o).__name__} not JSON serialisable')

def _json_dump_np(obj, fp, *args, default=None, **kwargs):
    if default is None:
        final_default = _np_default
    else:
        # Chain the defaults together
        final_default = lambda o: _np_default(o, default)
    return _json_dump_original(obj, fp, *args, default=final_default, **kwargs)

# Only patch if we're not already patched
if not hasattr(json.dump, '_np_patched'):
    json.dump = _json_dump_np
    json.dump._np_patched = True
# --- END PATCH ---

```

# 🌱 EchoSeed v3.0
Clean rebuild with:
- Drive persistence (chunks + master JSON)
- Single-thread guard + rate slider
- Lattice render capped at 1000 nodes
- Entropy graph
- No indentation headaches 🚀


```python
# 🔧 Core & External Libraries
import os, json, time, math, random, threading, collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.cluster import MiniBatchKMeans
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# 🧠 Sentence Embedding
!pip -q install sentence-transformers
from sentence_transformers import SentenceTransformer

# 📟 Jupyter + Widgets
from IPython.display import display, clear_output
import ipywidgets as widgets
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[5], line 5
          3 import numpy as np
          4 import networkx as nx
    ----> 5 import matplotlib.pyplot as plt
          6 from scipy.spatial.distance import cosine
          7 from sklearn.cluster import MiniBatchKMeans


    File /usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py:57
         55 from cycler import cycler  # noqa: F401
         56 import matplotlib
    ---> 57 import matplotlib.image
         58 from matplotlib import _api
         59 # Re-exported (import x as x) for typing.


    File /usr/local/lib/python3.11/dist-packages/matplotlib/image.py:25
         23 import matplotlib.artist as martist
         24 import matplotlib.colorizer as mcolorizer
    ---> 25 from matplotlib.backend_bases import FigureCanvasBase
         26 import matplotlib.colors as mcolors
         27 from matplotlib.transforms import (
         28     Affine2D, BboxBase, Bbox, BboxTransform, BboxTransformTo,
         29     IdentityTransform, TransformedBbox)


    File /usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:49
         46 import numpy as np
         48 import matplotlib as mpl
    ---> 49 from matplotlib import (
         50     _api, backend_tools as tools, cbook, colors, _docstring, text,
         51     _tight_bbox, transforms, widgets, is_interactive, rcParams)
         52 from matplotlib._pylab_helpers import Gcf
         53 from matplotlib.backend_managers import ToolManager


    File /usr/local/lib/python3.11/dist-packages/matplotlib/text.py:16
         14 from . import _api, artist, cbook, _docstring
         15 from .artist import Artist
    ---> 16 from .font_manager import FontProperties
         17 from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
         18 from .textpath import TextPath, TextToPath  # noqa # Logically located here


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1643
       1639     _log.info("generated new fontManager")
       1640     return fm
    -> 1643 fontManager = _load_fontmanager()
       1644 findfont = fontManager.findfont
       1645 get_font_names = fontManager.get_font_names


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1638, in _load_fontmanager(try_read_cache)
       1636             return fm
       1637 fm = FontManager()
    -> 1638 json_dump(fm, fm_path)
       1639 _log.info("generated new fontManager")
       1640 return fm


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1025, in json_dump(data, filename)
       1023 try:
       1024     with cbook._lock_path(filename), open(filename, 'w') as fh:
    -> 1025         json.dump(data, fh, cls=_JSONEncoder, indent=2)
       1026 except OSError as e:
       1027     _log.warning('Could not save font_manager cache %s', e)


    Cell In[4], line 24, in _json_dump_np(obj, fp, default, *args, **kwargs)
         21 else:
         22     # Chain the defaults together
         23     final_default = lambda o: _np_default(o, default)
    ---> 24 return _json_dump_original(obj, fp, *args, default=final_default, **kwargs)


    File /usr/lib/python3.11/json/__init__.py:179, in dump(obj, fp, skipkeys, ensure_ascii, check_circular, allow_nan, cls, indent, separators, default, sort_keys, **kw)
        173     iterable = cls(skipkeys=skipkeys, ensure_ascii=ensure_ascii,
        174         check_circular=check_circular, allow_nan=allow_nan, indent=indent,
        175         separators=separators,
        176         default=default, sort_keys=sort_keys, **kw).iterencode(obj)
        177 # could accelerate with writelines in some versions of Python, at
        178 # a debuggability cost
    --> 179 for chunk in iterable:
        180     fp.write(chunk)


    File /usr/lib/python3.11/json/encoder.py:439, in _make_iterencode.<locals>._iterencode(o, _current_indent_level)
        437         raise ValueError("Circular reference detected")
        438     markers[markerid] = o
    --> 439 o = _default(o)
        440 yield from _iterencode(o, _current_indent_level)
        441 if markers is not None:


    Cell In[4], line 16, in _np_default(o, _orig_default)
         14 if _orig_default is not None:
         15     return _orig_default(o)
    ---> 16 raise TypeError(f'{type(o).__name__} not JSON serialisable')


    TypeError: FontManager not JSON serialisable



```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
```

    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 2d3fff9b-fbd4-40fe-a096-966a9d805166)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/./modules.json


    Retrying in 1s [Retry 1/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: ce044e4a-85db-47c3-ad49-dfbac398470a)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/./modules.json


    Retrying in 2s [Retry 2/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 4f552ca6-abe5-4483-ac83-0d0413f86912)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/./modules.json


    Retrying in 4s [Retry 3/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: e0686dd5-717f-417c-b000-d363c7712c64)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/./modules.json


    Retrying in 8s [Retry 4/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 9693d2d2-2c48-4494-8677-e4ef25b842ed)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/./modules.json


    Retrying in 8s [Retry 5/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: a3f3fb3d-c896-4c67-98f9-3a9b4be7ba80)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/./modules.json


    No sentence-transformers model found with name sentence-transformers/all-MiniLM-L6-v2. Creating a new one with mean pooling.


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 0f1c19dc-c41d-4ffc-a20f-ac4f51acd7a0)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json


    Retrying in 1s [Retry 1/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 86bfb196-d165-465c-a8d0-de4e17ebaef3)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json


    Retrying in 2s [Retry 2/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 0db3691f-a80a-4e76-9de1-8b516703a8cb)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json


    Retrying in 4s [Retry 3/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 611df6db-a790-4c07-9552-1e3397e3b462)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json


    Retrying in 8s [Retry 4/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: a6dea91d-dc9c-4753-8050-6d9e62786439)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json


    Retrying in 8s [Retry 5/5].


    '(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 8c2dd05e-5bdd-4b56-b580-bb96173f5b38)')' thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json



    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    File ~/.local/lib/python3.11/site-packages/urllib3/connectionpool.py:773, in HTTPConnectionPool.urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, preload_content, decode_content, **response_kw)
        772 try:
    --> 773     self._prepare_proxy(conn)
        774 except (BaseSSLError, OSError, SocketTimeout) as e:


    File ~/.local/lib/python3.11/site-packages/urllib3/connectionpool.py:1042, in HTTPSConnectionPool._prepare_proxy(self, conn)
       1036 conn.set_tunnel(
       1037     scheme=tunnel_scheme,
       1038     host=self._tunnel_host,
       1039     port=self.port,
       1040     headers=self.proxy_headers,
       1041 )
    -> 1042 conn.connect()


    File ~/.local/lib/python3.11/site-packages/urllib3/connection.py:776, in HTTPSConnection.connect(self)
        774 self._has_connected_to_proxy = True
    --> 776 self._tunnel()
        777 # Override the host with the one we're requesting data from.


    File /usr/lib/python3.11/http/client.py:943, in HTTPConnection._tunnel(self)
        942     self.close()
    --> 943     raise OSError(f"Tunnel connection failed: {code} {message.strip()}")
        944 while True:


    OSError: Tunnel connection failed: 403 Forbidden

    
    The above exception was the direct cause of the following exception:


    ProxyError                                Traceback (most recent call last)

    ProxyError: ('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden'))

    
    The above exception was the direct cause of the following exception:


    MaxRetryError                             Traceback (most recent call last)

    File ~/.local/lib/python3.11/site-packages/requests/adapters.py:644, in HTTPAdapter.send(self, request, stream, timeout, verify, cert, proxies)
        643 try:
    --> 644     resp = conn.urlopen(
        645         method=request.method,
        646         url=url,
        647         body=request.body,
        648         headers=request.headers,
        649         redirect=False,
        650         assert_same_host=False,
        651         preload_content=False,
        652         decode_content=False,
        653         retries=self.max_retries,
        654         timeout=timeout,
        655         chunked=chunked,
        656     )
        658 except (ProtocolError, OSError) as err:


    File ~/.local/lib/python3.11/site-packages/urllib3/connectionpool.py:841, in HTTPConnectionPool.urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, preload_content, decode_content, **response_kw)
        839     new_e = ProtocolError("Connection aborted.", new_e)
    --> 841 retries = retries.increment(
        842     method, url, error=new_e, _pool=self, _stacktrace=sys.exc_info()[2]
        843 )
        844 retries.sleep()


    File ~/.local/lib/python3.11/site-packages/urllib3/util/retry.py:519, in Retry.increment(self, method, url, response, error, _pool, _stacktrace)
        518     reason = error or ResponseError(cause)
    --> 519     raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
        521 log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)


    MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))

    
    During handling of the above exception, another exception occurred:


    ProxyError                                Traceback (most recent call last)

    Cell In[6], line 2
          1 from sentence_transformers import SentenceTransformer
    ----> 2 model = SentenceTransformer("all-MiniLM-L6-v2")


    File /usr/local/lib/python3.11/dist-packages/sentence_transformers/SentenceTransformer.py:339, in SentenceTransformer.__init__(self, model_name_or_path, modules, device, prompts, default_prompt_name, similarity_fn_name, cache_folder, trust_remote_code, revision, local_files_only, token, use_auth_token, truncate_dim, model_kwargs, tokenizer_kwargs, config_kwargs, model_card_data, backend)
        327         modules, self.module_kwargs = self._load_sbert_model(
        328             model_name_or_path,
        329             token=token,
       (...)    336             config_kwargs=config_kwargs,
        337         )
        338     else:
    --> 339         modules = self._load_auto_model(
        340             model_name_or_path,
        341             token=token,
        342             cache_folder=cache_folder,
        343             revision=revision,
        344             trust_remote_code=trust_remote_code,
        345             local_files_only=local_files_only,
        346             model_kwargs=model_kwargs,
        347             tokenizer_kwargs=tokenizer_kwargs,
        348             config_kwargs=config_kwargs,
        349             has_modules=has_modules,
        350         )
        352 if modules is not None and not isinstance(modules, OrderedDict):
        353     modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])


    File /usr/local/lib/python3.11/dist-packages/sentence_transformers/SentenceTransformer.py:2112, in SentenceTransformer._load_auto_model(self, model_name_or_path, token, cache_folder, revision, trust_remote_code, local_files_only, model_kwargs, tokenizer_kwargs, config_kwargs, has_modules)
       2109 tokenizer_kwargs = shared_kwargs if tokenizer_kwargs is None else {**shared_kwargs, **tokenizer_kwargs}
       2110 config_kwargs = shared_kwargs if config_kwargs is None else {**shared_kwargs, **config_kwargs}
    -> 2112 transformer_model = Transformer(
       2113     model_name_or_path,
       2114     cache_dir=cache_folder,
       2115     model_args=model_kwargs,
       2116     tokenizer_args=tokenizer_kwargs,
       2117     config_args=config_kwargs,
       2118     backend=self.backend,
       2119 )
       2120 pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "mean")
       2121 if not local_files_only:


    File /usr/local/lib/python3.11/dist-packages/sentence_transformers/models/Transformer.py:88, in Transformer.__init__(self, model_name_or_path, max_seq_length, model_args, tokenizer_args, config_args, cache_dir, do_lower_case, tokenizer_name_or_path, backend)
         85 if config_args is None:
         86     config_args = {}
    ---> 88 config, is_peft_model = self._load_config(model_name_or_path, cache_dir, backend, config_args)
         89 self._load_model(model_name_or_path, config, cache_dir, backend, is_peft_model, **model_args)
         91 # Get the signature of the auto_model's forward method to pass only the expected arguments from `features`,
         92 # plus some common values like "input_ids", "attention_mask", etc.


    File /usr/local/lib/python3.11/dist-packages/sentence_transformers/models/Transformer.py:139, in Transformer._load_config(self, model_name_or_path, cache_dir, backend, config_args)
        123 def _load_config(
        124     self, model_name_or_path: str, cache_dir: str | None, backend: str, config_args: dict[str, Any]
        125 ) -> tuple[PeftConfig | PretrainedConfig, bool]:
        126     """Loads the transformers or PEFT configuration
        127 
        128     Args:
       (...)    136         tuple[PretrainedConfig, bool]: The model configuration and a boolean indicating whether the model is a PEFT model.
        137     """
        138     if (
    --> 139         find_adapter_config_file(
        140             model_name_or_path,
        141             cache_dir=cache_dir,
        142             token=config_args.get("token"),
        143             revision=config_args.get("revision"),
        144             local_files_only=config_args.get("local_files_only", False),
        145         )
        146         is not None
        147     ):
        148         if not is_peft_available():
        149             raise Exception(
        150                 "Loading a PEFT model requires installing the `peft` package. You can install it via `pip install peft`."
        151             )


    File /usr/local/lib/python3.11/dist-packages/transformers/utils/peft_utils.py:88, in find_adapter_config_file(model_id, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, _commit_hash)
         86         adapter_cached_filename = os.path.join(model_id, ADAPTER_CONFIG_NAME)
         87 else:
    ---> 88     adapter_cached_filename = cached_file(
         89         model_id,
         90         ADAPTER_CONFIG_NAME,
         91         cache_dir=cache_dir,
         92         force_download=force_download,
         93         resume_download=resume_download,
         94         proxies=proxies,
         95         token=token,
         96         revision=revision,
         97         local_files_only=local_files_only,
         98         subfolder=subfolder,
         99         _commit_hash=_commit_hash,
        100         _raise_exceptions_for_gated_repo=False,
        101         _raise_exceptions_for_missing_entries=False,
        102         _raise_exceptions_for_connection_errors=False,
        103     )
        105 return adapter_cached_filename


    File /usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py:322, in cached_file(path_or_repo_id, filename, **kwargs)
        264 def cached_file(
        265     path_or_repo_id: Union[str, os.PathLike],
        266     filename: str,
        267     **kwargs,
        268 ) -> Optional[str]:
        269     """
        270     Tries to locate a file in a local folder and repo, downloads and cache it if necessary.
        271 
       (...)    320     ```
        321     """
    --> 322     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
        323     file = file[0] if file is not None else file
        324     return file


    File /usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py:567, in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
        564     # Any other Exception type should now be re-raised, in order to provide helpful error messages and break the execution flow
        565     # (EntryNotFoundError will be treated outside this block and correctly re-raised if needed)
        566     elif not isinstance(e, EntryNotFoundError):
    --> 567         raise e
        569 resolved_files = [
        570     _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
        571 ]
        572 # If there are any missing file and the flag is active, raise


    File /usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py:479, in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
        476 try:
        477     if len(full_filenames) == 1:
        478         # This is slightly better for only 1 file
    --> 479         hf_hub_download(
        480             path_or_repo_id,
        481             filenames[0],
        482             subfolder=None if len(subfolder) == 0 else subfolder,
        483             repo_type=repo_type,
        484             revision=revision,
        485             cache_dir=cache_dir,
        486             user_agent=user_agent,
        487             force_download=force_download,
        488             proxies=proxies,
        489             resume_download=resume_download,
        490             token=token,
        491             local_files_only=local_files_only,
        492         )
        493     else:
        494         snapshot_download(
        495             path_or_repo_id,
        496             allow_patterns=full_filenames,
       (...)    505             local_files_only=local_files_only,
        506         )


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py:114, in validate_hf_hub_args.<locals>._inner_fn(*args, **kwargs)
        111 if check_use_auth_token:
        112     kwargs = smoothly_deprecate_use_auth_token(fn_name=fn.__name__, has_token=has_token, kwargs=kwargs)
    --> 114 return fn(*args, **kwargs)


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py:1007, in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
        987     return _hf_hub_download_to_local_dir(
        988         # Destination
        989         local_dir=local_dir,
       (...)   1004         local_files_only=local_files_only,
       1005     )
       1006 else:
    -> 1007     return _hf_hub_download_to_cache_dir(
       1008         # Destination
       1009         cache_dir=cache_dir,
       1010         # File info
       1011         repo_id=repo_id,
       1012         filename=filename,
       1013         repo_type=repo_type,
       1014         revision=revision,
       1015         # HTTP info
       1016         endpoint=endpoint,
       1017         etag_timeout=etag_timeout,
       1018         headers=hf_headers,
       1019         proxies=proxies,
       1020         token=token,
       1021         # Additional options
       1022         local_files_only=local_files_only,
       1023         force_download=force_download,
       1024     )


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py:1070, in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
       1066         return pointer_path
       1068 # Try to get metadata (etag, commit_hash, url, size) from the server.
       1069 # If we can't, a HEAD request error is returned.
    -> 1070 (url_to_download, etag, commit_hash, expected_size, xet_file_data, head_call_error) = _get_metadata_or_catch_error(
       1071     repo_id=repo_id,
       1072     filename=filename,
       1073     repo_type=repo_type,
       1074     revision=revision,
       1075     endpoint=endpoint,
       1076     proxies=proxies,
       1077     etag_timeout=etag_timeout,
       1078     headers=headers,
       1079     token=token,
       1080     local_files_only=local_files_only,
       1081     storage_folder=storage_folder,
       1082     relative_filename=relative_filename,
       1083 )
       1085 # etag can be None for several reasons:
       1086 # 1. we passed local_files_only.
       1087 # 2. we don't have a connection
       (...)   1093 # If the specified revision is a commit hash, look inside "snapshots".
       1094 # If the specified revision is a branch or tag, look inside "refs".
       1095 if head_call_error is not None:
       1096     # Couldn't make a HEAD call => let's try to find a local file


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py:1543, in _get_metadata_or_catch_error(repo_id, filename, repo_type, revision, endpoint, proxies, etag_timeout, headers, token, local_files_only, relative_filename, storage_folder)
       1541 try:
       1542     try:
    -> 1543         metadata = get_hf_file_metadata(
       1544             url=url, proxies=proxies, timeout=etag_timeout, headers=headers, token=token, endpoint=endpoint
       1545         )
       1546     except EntryNotFoundError as http_error:
       1547         if storage_folder is not None and relative_filename is not None:
       1548             # Cache the non-existence of the file


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py:114, in validate_hf_hub_args.<locals>._inner_fn(*args, **kwargs)
        111 if check_use_auth_token:
        112     kwargs = smoothly_deprecate_use_auth_token(fn_name=fn.__name__, has_token=has_token, kwargs=kwargs)
    --> 114 return fn(*args, **kwargs)


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py:1460, in get_hf_file_metadata(url, token, proxies, timeout, library_name, library_version, user_agent, headers, endpoint)
       1457 hf_headers["Accept-Encoding"] = "identity"  # prevent any compression => we want to know the real size of the file
       1459 # Retrieve metadata
    -> 1460 r = _request_wrapper(
       1461     method="HEAD",
       1462     url=url,
       1463     headers=hf_headers,
       1464     allow_redirects=False,
       1465     follow_relative_redirects=True,
       1466     proxies=proxies,
       1467     timeout=timeout,
       1468 )
       1469 hf_raise_for_status(r)
       1471 # Return


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py:283, in _request_wrapper(method, url, follow_relative_redirects, **params)
        281 # Recursively follow relative redirects
        282 if follow_relative_redirects:
    --> 283     response = _request_wrapper(
        284         method=method,
        285         url=url,
        286         follow_relative_redirects=False,
        287         **params,
        288     )
        290     # If redirection, we redirect only relative paths.
        291     # This is useful in case of a renamed repository.
        292     if 300 <= response.status_code <= 399:


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py:306, in _request_wrapper(method, url, follow_relative_redirects, **params)
        303     return response
        305 # Perform request and return if status_code is not in the retry list.
    --> 306 response = http_backoff(method=method, url=url, **params)
        307 hf_raise_for_status(response)
        308 return response


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py:325, in http_backoff(method, url, max_retries, base_wait_time, max_wait_time, retry_on_exceptions, retry_on_status_codes, **kwargs)
        322         reset_sessions()  # In case of SSLError it's best to reset the shared requests.Session objects
        324     if nb_tries > max_retries:
    --> 325         raise err
        327 # Sleep for X seconds
        328 logger.warning(f"Retrying in {sleep_time}s [Retry {nb_tries}/{max_retries}].")


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py:306, in http_backoff(method, url, max_retries, base_wait_time, max_wait_time, retry_on_exceptions, retry_on_status_codes, **kwargs)
        303     kwargs["data"].seek(io_obj_initial_pos)
        305 # Perform request and return if status_code is not in the retry list.
    --> 306 response = session.request(method=method, url=url, **kwargs)
        307 if response.status_code not in retry_on_status_codes:
        308     return response


    File ~/.local/lib/python3.11/site-packages/requests/sessions.py:589, in Session.request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        584 send_kwargs = {
        585     "timeout": timeout,
        586     "allow_redirects": allow_redirects,
        587 }
        588 send_kwargs.update(settings)
    --> 589 resp = self.send(prep, **send_kwargs)
        591 return resp


    File ~/.local/lib/python3.11/site-packages/requests/sessions.py:703, in Session.send(self, request, **kwargs)
        700 start = preferred_clock()
        702 # Send the request
    --> 703 r = adapter.send(request, **kwargs)
        705 # Total elapsed time of the request (approximately)
        706 elapsed = preferred_clock() - start


    File /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py:95, in UniqueRequestIdAdapter.send(self, request, *args, **kwargs)
         93     logger.debug(f"Send: {_curlify(request)}")
         94 try:
    ---> 95     return super().send(request, *args, **kwargs)
         96 except requests.RequestException as e:
         97     request_id = request.headers.get(X_AMZN_TRACE_ID)


    File ~/.local/lib/python3.11/site-packages/requests/adapters.py:671, in HTTPAdapter.send(self, request, stream, timeout, verify, cert, proxies)
        668     raise RetryError(e, request=request)
        670 if isinstance(e.reason, _ProxyError):
    --> 671     raise ProxyError(e, request=request)
        673 if isinstance(e.reason, _SSLError):
        674     # This branch is for urllib3 v1.22 and later.
        675     raise SSLError(e, request=request)


    ProxyError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: 8c2dd05e-5bdd-4b56-b580-bb96173f5b38)')



```python
# ===== CORE DATA STRUCTURES =====
glyph_log = []
id2glyph = {}
glyph_graph = nx.Graph()
graph_lock = threading.Lock()
glyph_id = 0
running = threading.Event()
seen_signatures = set()
reflex_free = []
stagnant_counter = {}
conceptual_attractors = {}  # Track hierarchical meaning emergence
temporal_clusters = {}  # Track temporal synchronization patterns
dormant_glyphs = {}  # Glyphs in dormant state

# ===== ENHANCED CONSTANTS =====
MAX_RAM_GLYPHS = 20_000
LOW_ENTROPY = 100
LOW_KEEP_PROB = 0.025
PRUNE_AFTER = 250
ATTRACTOR_THRESHOLD = 5  # Minimum offspring to become attractor
DORMANCY_THRESHOLD = 500  # Generations before dormancy consideration
REACTIVATION_PROBABILITY = 0.1  # Chance dormant glyph reactivates per cycle

# ===== REFLEX SYSTEM EXPANSION =====
class ReflexType(Enum):
    DEFENSIVE = "defensive"    # Preserve existing patterns
    EXPLORATORY = "exploratory"  # Seek novel combinations
    COLLABORATIVE = "collaborative"  # Bridge disparate concepts
    CONSOLIDATIVE = "consolidative"  # Strengthen existing connections
    METAMORPHIC = "metamorphic"  # Transform fundamental structure

@dataclass
class ReflexProfile:
    primary_type: ReflexType
    intensity: float  # 0.0 to 1.0
    trigger_conditions: List[str]
    activation_count: int = 0
    last_activation: int = 0

# ===== SEASONAL DYNAMICS =====
class SeasonalPhase(Enum):
    EXPLORATION = "exploration"
    CONSOLIDATION = "consolidation"
    DORMANCY = "dormancy"
    RENAISSANCE = "renaissance"

generation_count = 0
current_season = SeasonalPhase.EXPLORATION
season_duration = 1000  # Generations per season
season_counter = 0

# ===== SEMANTIC EMBEDDING HELPERS =====
def tag_vec(tag):
    return model.encode(tag)

def semantic_novelty(vec):
    if not glyph_log:
        return 1.0
    # Enhanced novelty: consider temporal context
    recent_sample = [g for g in glyph_log[-500:] if g is not None]
    if not recent_sample:
        return 1.0

    sample = random.sample(recent_sample, min(100, len(recent_sample)))
    dists = [cosine(vec, g['vec']) for g in sample]
    return max(dists) if dists else 1.0

def productive_novelty(g):
    """Novelty weighted by offspring quality and diversity."""
    base_novelty = semantic_novelty(g['vec'])

    # Bonus for productive ancestors
    ancestry_bonus = 0
    for ancestor_id in g.get('ancestry', []):
        ancestor = id2glyph.get(ancestor_id)
        if ancestor and ancestor['id'] in conceptual_attractors:
            ancestry_bonus += 0.1

    return base_novelty + ancestry_bonus

def embed(tags):
    vs = [tag_vec(t) for t in tags]
    return np.mean(vs, axis=0)

# ===== ENHANCED INFLUENCE CALCULATION =====
def calculate_influence(g):
    """Calculate multi-dimensional influence including temporal and semantic factors."""
    if g['id'] not in glyph_graph:
        return 0

    # Get all children
    children = []
    for neighbor in glyph_graph.neighbors(g['id']):
        neighbor_glyph = id2glyph.get(neighbor)
        if neighbor_glyph and g['id'] in neighbor_glyph.get('ancestry', []):
            children.append(neighbor_glyph)

    if not children:
        return 0

    # Original metrics
    offspring_novelty = sum(productive_novelty(child) for child in children) / len(children)
    tag_diversity = len(set().union(*[child['tags'] for child in children])) / max(1, len(children))
    max_depth = max((calculate_cascade_depth(child, visited=set()) for child in children), default=0)

    # New metrics
    temporal_coherence = calculate_temporal_coherence(g, children)
    cross_pollination_factor = calculate_cross_pollination(g)

    return (offspring_novelty * 0.35 +
            tag_diversity * 0.25 +
            max_depth * 0.15 +
            temporal_coherence * 0.15 +
            cross_pollination_factor * 0.10)

def calculate_temporal_coherence(parent, children):
    """Measure how well a glyph's influence persists across generations."""
    if not children:
        return 0

    # Check if children spawn their own successful offspring
    successful_children = 0
    for child in children:
        child_children = [id2glyph.get(n) for n in glyph_graph.neighbors(child['id'])
                         if id2glyph.get(n) and child['id'] in id2glyph.get(n).get('ancestry', [])]
        if child_children:
            successful_children += 1

    return successful_children / len(children)

def calculate_cross_pollination(g):
    """Measure how often this glyph bridges different semantic clusters."""
    if g['id'] not in glyph_graph:
        return 0

    neighbors = [id2glyph.get(n) for n in glyph_graph.neighbors(g['id']) if id2glyph.get(n)]
    if len(neighbors) < 2:
        return 0

    # Calculate semantic distances between neighbors
    distances = []
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            dist = cosine(neighbors[i]['vec'], neighbors[j]['vec'])
            distances.append(dist)

    return np.mean(distances) if distances else 0

def calculate_cascade_depth(g, visited=None, max_depth=7):
    """Enhanced cascade depth with cycle detection."""
    if visited is None:
        visited = set()

    if g['id'] in visited or max_depth <= 0:
        return 0

    visited.add(g['id'])

    children = []
    for neighbor in glyph_graph.neighbors(g['id']):
        neighbor_glyph = id2glyph.get(neighbor)
        if neighbor_glyph and g['id'] in neighbor_glyph.get('ancestry', []):
            children.append(neighbor_glyph)

    if not children:
        return 1

    return 1 + max(calculate_cascade_depth(child, visited.copy(), max_depth-1) for child in children)

# ===== CONCEPTUAL ATTRACTOR DETECTION =====
def update_conceptual_attractors():
    """Detect glyphs that consistently appear in high-influence lineages."""
    global conceptual_attractors

    influence_rankings = {}
    for glyph_id, glyph in list(id2glyph.items()):
        influence = calculate_influence(glyph)
        if influence > 0:
            influence_rankings[glyph_id] = influence

    # Find glyphs that appear frequently in ancestry of high-influence glyphs
    top_influencers = sorted(influence_rankings.items(), key=lambda x: x[1], reverse=True)[:50]

    ancestry_counts = collections.Counter()
    for glyph_id, _ in top_influencers:
        glyph = id2glyph.get(glyph_id)
        if glyph:
            for ancestor_id in glyph.get('ancestry', []):
                ancestry_counts[ancestor_id] += 1

    # Update attractors
    for ancestor_id, count in ancestry_counts.items():
        if count >= ATTRACTOR_THRESHOLD:
            if ancestor_id not in conceptual_attractors:
                conceptual_attractors[ancestor_id] = {
                    'discovered_generation': generation_count,
                    'influence_episodes': []
                }
            conceptual_attractors[ancestor_id]['influence_episodes'].append(generation_count)

# ===== ENHANCED REFLEX SYSTEM =====
def determine_reflex_type(g, context_glyphs):
    """Determine appropriate reflex type based on glyph and context."""
    entropy_ratio = g['entropy'] / max(1, np.mean([cg['entropy'] for cg in context_glyphs]))
    semantic_isolation = semantic_novelty(g['vec'])

    if entropy_ratio < 0.5 and semantic_isolation < 0.3:
        return ReflexType.DEFENSIVE
    elif semantic_isolation > 0.8:
        return ReflexType.EXPLORATORY
    elif len(g['tags']) > 3 and entropy_ratio > 1.2:
        return ReflexType.COLLABORATIVE
    elif g['id'] in conceptual_attractors:
        return ReflexType.CONSOLIDATIVE
    else:
        return ReflexType.METAMORPHIC

def create_reflex_glyph(parent, reflex_type: ReflexType):
    """Create specialized reflex glyph based on type."""
    base_tags = parent['tags'].copy()

    if reflex_type == ReflexType.DEFENSIVE:
        new_tags = base_tags + ['reflex', 'preserve', 'stable']
    elif reflex_type == ReflexType.EXPLORATORY:
        new_tags = base_tags + ['reflex', 'seek', 'novel', random_tag()]
    elif reflex_type == ReflexType.COLLABORATIVE:
        # Find distant glyph to bridge with
        distant_glyph = find_semantically_distant_glyph(parent)
        if distant_glyph:
            new_tags = base_tags + distant_glyph['tags'][:2] + ['reflex', 'bridge']
        else:
            new_tags = base_tags + ['reflex', 'bridge']
    elif reflex_type == ReflexType.CONSOLIDATIVE:
        new_tags = base_tags + ['reflex', 'strengthen', 'anchor']
    else:  # METAMORPHIC
        new_tags = [mutate_tag(base_tags)] + ['reflex', 'transform', 'evolve']

    new_g = {
        'id': gen_id(),
        'tags': list(set(new_tags)),
        'entropy': 0,
        'ancestry': [parent['id']],
        'reflex_profile': ReflexProfile(
            primary_type=reflex_type,
            intensity=random.uniform(0.5, 1.0),
            trigger_conditions=[f"entropy<{parent['entropy']*1.2}", "unknown_tag"],
            activation_count=1,
            last_activation=generation_count
        )
    }
    new_g['entropy'] = calc_entropy(new_g)
    return new_g

def find_semantically_distant_glyph(reference_glyph):
    """Find a glyph that's semantically distant from the reference."""
    if len(glyph_log) < 10:
        return None

    candidates = random.sample(glyph_log, min(50, len(glyph_log)))
    distances = [(cosine(reference_glyph['vec'], g['vec']), g) for g in candidates]
    distances.sort(key=lambda x: x[0], reverse=True)

    return distances[0][1] if distances else None

# ===== SEASONAL SYSTEM =====
class SeasonalPhase(Enum):
    EXPLORATION = "exploration"
    CONSOLIDATION = "consolidation"
    DORMANCY = "dormancy"
    RENAISSANCE = "renaissance"

def update_seasonal_phase():
    """Update seasonal phase affecting system behavior."""
    global current_season, season_counter

    season_counter += 1
    if season_counter >= season_duration:
        season_counter = 0
        seasons = list(SeasonalPhase)
        current_index = seasons.index(current_season)
        current_season = seasons[(current_index + 1) % len(seasons)]

        print(f"Season change: {current_season.value} (Generation {generation_count})")

        # Seasonal effects
        if current_season == SeasonalPhase.DORMANCY:
            activate_dormancy_phase()
        elif current_season == SeasonalPhase.RENAISSANCE:
            activate_renaissance_phase()

def activate_dormancy_phase():
    """Move low-activity glyphs to dormant state."""
    global dormant_glyphs

    for glyph_id, glyph in list(id2glyph.items()):
        if (stagnant_counter.get(glyph_id, 0) > DORMANCY_THRESHOLD and
            glyph_id not in conceptual_attractors and
            calculate_influence(glyph) < 0.1):

            dormant_glyphs[glyph_id] = {
                'glyph': glyph,
                'dormancy_start': generation_count,
                'reactivation_potential': random.uniform(0.1, 0.8)
            }

            # Remove from active systems
            del id2glyph[glyph_id]
            if glyph in glyph_log:
                glyph_log.remove(glyph)

def activate_renaissance_phase():
    """Reactivate dormant glyphs with new perspectives."""
    global dormant_glyphs

    to_reactivate = []
    for glyph_id, dormant_info in dormant_glyphs.items():
        if random.random() < dormant_info['reactivation_potential'] * REACTIVATION_PROBABILITY:
            to_reactivate.append(glyph_id)

    for glyph_id in to_reactivate:
        dormant_info = dormant_glyphs.pop(glyph_id)
        glyph = dormant_info['glyph']

        # Enhance glyph with renaissance perspective
        glyph['tags'].append('renaissance')
        glyph['entropy'] = calc_entropy(glyph)  # Recalculate with current generation
        glyph['vec'] = embed(glyph['tags'])

        # Reintroduce to active systems
        id2glyph[glyph_id] = glyph
        glyph_log.append(glyph)

        print(f"Reactivated: {glyph_id} with tags {glyph['tags']}")

# ===== CORE SYSTEM FUNCTIONS =====
id_lock = threading.Lock()
def gen_id():
    global glyph_id
    with id_lock:
        glyph_id += 1
        return f"g{glyph_id:04}"

def random_tag():
    base_tags = ['origin', 'flex', 'ghost', 'fractal', 'wild', 'mirror', 'unknown', 'stable']
    seasonal_tags = {
        SeasonalPhase.EXPLORATION: ['pioneer', 'venture', 'discover'],
        SeasonalPhase.CONSOLIDATION: ['anchor', 'strengthen', 'unify'],
        SeasonalPhase.DORMANCY: ['rest', 'potential', 'dormant'],
        SeasonalPhase.RENAISSANCE: ['reborn', 'transformed', 'awakened']
    }

    all_tags = base_tags + seasonal_tags.get(current_season, [])
    return random.choice(all_tags)

def calc_entropy(g, gen=None):
    if gen is None:
        gen = generation_count

    # Base entropy
    base = len(g['tags']) * 42 + random.randint(0, 58)

    # Generational pressure
    gen_pressure = gen * 10

    # Seasonal modifier
    seasonal_modifier = {
        SeasonalPhase.EXPLORATION: 1.2,
        SeasonalPhase.CONSOLIDATION: 0.8,
        SeasonalPhase.DORMANCY: 0.6,
        SeasonalPhase.RENAISSANCE: 1.5
    }

    return int(base + gen_pressure * seasonal_modifier.get(current_season, 1.0))

def fitness(g):
    sem = productive_novelty(g)
    influence = calculate_influence(g)

    # Seasonal fitness adjustments
    seasonal_bonus = 0
    if current_season == SeasonalPhase.EXPLORATION and sem > 0.7:
        seasonal_bonus = 50
    elif current_season == SeasonalPhase.CONSOLIDATION and g['id'] in conceptual_attractors:
        seasonal_bonus = 75
    elif current_season == SeasonalPhase.RENAISSANCE and 'renaissance' in g['tags']:
        seasonal_bonus = 100

    return 200 * sem - g['entropy'] + 50 * influence + seasonal_bonus

def mutate_tag(tags):
    if len(tags) < 2:
        return 'm_misc'
    a, b = random.sample(tags, 2)

    # Enhanced mutation with semantic awareness
    if current_season == SeasonalPhase.EXPLORATION:
        return f"{a}→{b}"  # Directional mutation
    elif current_season == SeasonalPhase.CONSOLIDATION:
        return f"{a}∧{b}"  # Conjunction
    else:
        return f"{a}+{b}"  # Classic combination

def signature(g, bucket=25):
    return (tuple(sorted(g['tags'])), g['entropy'] // bucket)

def is_novel(g):
    if g['entropy'] < LOW_ENTROPY:
        return random.random() < LOW_KEEP_PROB

    sig = signature(g)
    if sig in seen_signatures:
        return False

    seen_signatures.add(sig)
    return True

log_lock = threading.Lock()
def maybe_store(g):
    g['vec'] = embed(g['tags'])
    if is_novel(g):
        with log_lock:
            glyph_log.append(g)
            id2glyph[g['id']] = g
        with graph_lock:
            glyph_graph.add_node(g['id'], entropy=g['entropy'], tags=g['tags'], fit=fitness(g))
            stagnant_counter[g['id']] = 0
        if 'reflex' not in g['tags']:
            with log_lock:
                reflex_free.append(g)

def create_glyph():
    g = {
        'id': gen_id(),
        'tags': list({random_tag() for _ in range(random.randint(1, 3))}),
        'entropy': 0,
        'ancestry': []
    }
    g['entropy'] = calc_entropy(g)
    g['vec'] = embed(g['tags'])
    return g

# ===== ENHANCED COLLISION SYSTEM =====
def smart_edge_explosion(child, parents):
    """Intelligent edge creation based on semantic similarity and temporal patterns."""
    candidates = [g for g in glyph_log if g['id'] not in {p['id'] for p in parents} and g['id'] != child['id']]

    if not candidates:
        return

    # Semantic similarity connections
    semantic_connections = []
    for candidate in candidates:
        similarity = 1 - cosine(child['vec'], candidate['vec'])
        if 0.3 < similarity < 0.8:  # Sweet spot for productive connections
            semantic_connections.append((candidate, similarity))

    # Temporal synchronization connections
    temporal_connections = []
    current_gen = generation_count
    for candidate in candidates:
        if candidate['id'] in temporal_clusters:
            gen_diff = abs(temporal_clusters[candidate['id']] - current_gen)
            if gen_diff < 50:  # Same temporal cluster
                temporal_connections.append(candidate)

    # Conceptual attractor connections
    attractor_connections = [c for c in candidates if c['id'] in conceptual_attractors]

    # Create connections with weighted selection
    all_connections = []
    all_connections.extend([(c[0], c[1] * 2) for c in semantic_connections[:5]])  # Top 5 semantic
    all_connections.extend([(c, 1.5) for c in temporal_connections[:3]])         # Top 3 temporal
    all_connections.extend([(c, 3.0) for c in attractor_connections[:2]])        # Top 2 attractors

    # Add some random connections for exploration
    random_candidates = random.sample(candidates, min(3, len(candidates)))
    all_connections.extend([(c, 0.5) for c in random_candidates])

    # Create edges
    with graph_lock:
        for candidate, weight in all_connections[:15]:  # Limit total connections
            glyph_graph.add_edge(child['id'], candidate['id'], weight=weight)
            stagnant_counter[candidate['id']] = 0

def collide(a, b):
    """Enhanced collision with reflex system integration."""
    child_tags = {*a['tags'], *b['tags'], mutate_tag(a['tags'] + b['tags'])}
    child = {
        'id': gen_id(),
        'tags': list(child_tags),
        'entropy': 0,
        'ancestry': [a['id'], b['id']],
        'generation_born': generation_count
    }
    child['entropy'] = calc_entropy(child)

    maybe_store(child)

    with graph_lock:
        glyph_graph.add_edge(a['id'], child['id'])
        glyph_graph.add_edge(b['id'], child['id'])
        stagnant_counter[a['id']] = stagnant_counter[b['id']] = stagnant_counter[child['id']] = 0

    # Enhanced edge explosion
    smart_edge_explosion(child, [a, b])

    # Update temporal clusters
    temporal_clusters[child['id']] = generation_count

def select_parents(k=10):
    """Enhanced parent selection with seasonal and attractor weighting."""
    with graph_lock:
        nodes = list(glyph_graph.nodes(data=True))

    if len(nodes) < 2:
        return []

    weights = []
    for node_id, node_data in nodes:
        base_weight = max(1, node_data.get('fit', 1))

        # Attractor bonus
        if node_id in conceptual_attractors:
            base_weight *= 2

        # Seasonal weighting
        glyph = id2glyph.get(node_id)
        if glyph:
            if current_season == SeasonalPhase.EXPLORATION and semantic_novelty(glyph['vec']) > 0.5:
                base_weight *= 1.5
            elif current_season == SeasonalPhase.CONSOLIDATION and len(glyph['tags']) > 2:
                base_weight *= 1.3

        weights.append(base_weight)

    chosen = random.choices(nodes, weights=weights, k=min(k, len(nodes)))
    return [n[0] for n in chosen]

def batch_collide(max_pairs=30):
    parents = select_parents(max_pairs)
    if len(parents) < 2:
        return

    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            a = id2glyph.get(parents[i])
            b = id2glyph.get(parents[j])
            if a and b:
                collide(a, b)

# ===== ENHANCED REFLEX SYSTEM =====
def reflex_test():
    """Enhanced reflex system with typed responses."""
    context_glyphs = glyph_log[-100:] if len(glyph_log) > 100 else glyph_log

    for g in [g for g in glyph_log if 'unknown' in g['tags'] and g['entropy'] < 150]:
        reflex_type = determine_reflex_type(g, context_glyphs)

        # Create multiple reflex glyphs based on type
        reflex_count = 3 if reflex_type == ReflexType.EXPLORATORY else 2

        for _ in range(reflex_count):
            new_g = create_reflex_glyph(g, reflex_type)
            maybe_store(new_g)

            with graph_lock:
                glyph_graph.add_edge(g['id'], new_g['id'])

            # Reflex collision with parent
            collide(g, new_g)

def prune_stagnant():
    """Enhanced pruning with dormancy system."""
    to_remove = []
    to_dormant = []

    with graph_lock:
        for node in list(glyph_graph.nodes):
            stagnant_counter[node] += 1

            if stagnant_counter[node] > PRUNE_AFTER:
                if (current_season == SeasonalPhase.DORMANCY and
                    node not in conceptual_attractors and
                    stagnant_counter[node] < DORMANCY_THRESHOLD):
                    to_dormant.append(node)
                else:
                    to_remove.append(node)

        # Remove nodes
        for n in to_remove:
            glyph_graph.remove_node(n)
            stagnant_counter.pop(n, None)

    # Handle dormancy
    for node_id in to_dormant:
        glyph = id2glyph.get(node_id)
        if glyph:
            dormant_glyphs[node_id] = {
                'glyph': glyph,
                'dormancy_start': generation_count,
                'reactivation_potential': random.uniform(0.2, 0.9)
            }

    # Clean up data structures
    for nid in to_remove:
        id2glyph.pop(nid, None)

    ids_to_remove = set(to_remove)
    glyph_log[:] = [g for g in glyph_log if g['id'] not in ids_to_remove]

def rewire_shortcuts(pct=0.05):
    """Enhanced rewiring with semantic awareness."""
    with graph_lock:
        nodes = list(glyph_graph.nodes)

    if len(nodes) < 3:
        return

    sample = random.sample(nodes, max(2, int(len(nodes) * pct)))

    for nid in sample:
        source_glyph = id2glyph.get(nid)
        if not source_glyph:
            continue

        # Find semantically interesting targets
        candidates = [n for n in nodes if n != nid]

        if not candidates:
            continue

        target_weights = []
        for candidate in candidates:
            target_glyph = id2glyph.get(candidate)
            if target_glyph:
                # Weight by semantic distance and novelty
                distance = cosine(source_glyph['vec'], target_glyph['vec'])
                novelty = semantic_novelty(target_glyph['vec'])
                weight = distance * 0.7 + novelty * 0.3
                target_weights.append(weight)
            else:
                target_weights.append(0.1)

        if target_weights:
            target = random.choices(candidates, weights=target_weights)[0]
            with graph_lock:
                glyph_graph.add_edge(nid, target)
                stagnant_counter[nid] = 0
                stagnant_counter[target] = 0

# ===== ENHANCED PERSISTENCE =====


def save_logs(chunk_size: int = 250) -> None:
    """Dump recent glyphs, full log, and reflex-free list to JSON in LOG_DIR."""
    import os, json, numpy as np, pathlib

    def _jsonify(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, (list, tuple, set)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()
                    if k not in ('graph', 'nx_obj')}
        return repr(obj)

    pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    idx = len([f for f in os.listdir(LOG_DIR)
               if f.startswith('chunk_') and f.endswith('.json')])

    recent = glyph_log[-chunk_size:] if len(glyph_log) >= chunk_size else glyph_log

    with open(f"{LOG_DIR}/chunk_{idx:04}.json", "w", encoding="utf-8") as f:
        json.dump([_jsonify(g) for g in recent], f, separators=(',', ':'), ensure_ascii=False)

    with open(f"{LOG_DIR}/master_log.json", "w", encoding="utf-8") as f:
        json.dump([_jsonify(g) for g in glyph_log], f, separators=(',', ':'), ensure_ascii=False)

    with open(f"{LOG_DIR}/reflex_free.json", "w", encoding="utf-8") as f:
        json.dump([_jsonify(g) for g in reflex_free], f, separators=(',', ':'), ensure_ascii=False)

    print(f"✅ Saved to {LOG_DIR} (chunk_{idx:04}.json, master_log.json, reflex_free.json)")

def generate():
    """Enhanced generation loop with seasonal dynamics."""
    global generation_count

    while running.is_set():
        generation_count += 1

        # Update seasonal phase
        update_seasonal_phase()

        # Create new glyph
        g = create_glyph()
        maybe_store(g)

        # Batch operations
        batch_collide()
        reflex_test()

        # Periodic saves with metadata
        if len(glyph_log) % 250 == 0 and len(glyph_log) > 0:
            save_logs()

        # Enhanced cluster analysis with temporal tracking
        if generation_count % 500 == 0 and len(glyph_log) >= 500:
            perform_deep_analysis()

        # Adaptive sleep based on seasonal phase
        sleep_duration = {
            SeasonalPhase.EXPLORATION: 0.1,
            SeasonalPhase.CONSOLIDATION: 0.2,
            SeasonalPhase.DORMANCY: 0.5,
            SeasonalPhase.RENAISSANCE: 0.05
        }
        time.sleep(sleep_duration.get(current_season, 0.2))

# ===== DEEP ANALYSIS SYSTEM =====
def perform_deep_analysis():
    """Comprehensive system analysis with emergent pattern detection."""
    print(f"\n=== Deep Analysis: Generation {generation_count} ===")

    # Semantic clustering with temporal dimension
    if len(glyph_log) >= 500:
        analyze_semantic_clusters()

    # Conceptual attractor analysis
    analyze_conceptual_attractors()

    # Network topology analysis
    analyze_network_topology()

    # Seasonal transition analysis
    analyze_seasonal_patterns()

    # Emergence detection
    detect_emergent_patterns()

def analyze_semantic_clusters():
    """Advanced semantic clustering with temporal tracking."""
    sample_size = min(2000, len(glyph_log))
    sample_glyphs = random.sample(glyph_log, sample_size)

    # Create embeddings matrix
    embeddings = np.stack([g['vec'] for g in sample_glyphs])

    # Temporal clustering (generation-based)
    generations = np.array([g.get('generation_born', 0) for g in sample_glyphs])

    # Multi-dimensional clustering
    n_clusters = min(25, len(sample_glyphs) // 20)
    semantic_clusters = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256).fit(embeddings)

    # Analyze cluster evolution
    cluster_evolution = collections.defaultdict(list)
    for i, (glyph, cluster_id) in enumerate(zip(sample_glyphs, semantic_clusters.labels_)):
        cluster_evolution[cluster_id].append({
            'glyph_id': glyph['id'],
            'generation': glyph.get('generation_born', 0),
            'tags': glyph['tags'],
            'entropy': glyph['entropy']
        })

    # Identify dominant themes per cluster
    cluster_themes = {}
    for cluster_id, glyphs in cluster_evolution.items():
        all_tags = [tag for g in glyphs for tag in g['tags']]
        tag_counts = collections.Counter(all_tags)
        dominant_tags = [tag for tag, count in tag_counts.most_common(3)]

        avg_entropy = np.mean([g['entropy'] for g in glyphs])
        gen_span = max(g['generation'] for g in glyphs) - min(g['generation'] for g in glyphs)

        cluster_themes[cluster_id] = {
            'dominant_tags': dominant_tags,
            'avg_entropy': avg_entropy,
            'generation_span': gen_span,
            'size': len(glyphs)
        }

    # Save cluster analysis
    with open('glyph_data/cluster_analysis.json', 'w') as f:
        json.dump(cluster_themes, f, indent=2, default=_jsonify)

    print(f"Analyzed {n_clusters} semantic clusters")
    for cid, theme in list(cluster_themes.items())[:5]:
        print(f"  Cluster {cid}: {theme['dominant_tags']} (size: {theme['size']}, entropy: {theme['avg_entropy']:.1f})")

def analyze_conceptual_attractors():
    """Analyze the evolution and influence of conceptual attractors."""
    if not conceptual_attractors:
        print("No conceptual attractors detected yet")
        return

    print(f"\nConceptual Attractors ({len(conceptual_attractors)}):")

    attractor_analysis = {}
    for attractor_id, info in conceptual_attractors.items():
        attractor_glyph = id2glyph.get(attractor_id)
        if not attractor_glyph:
            continue

        # Calculate lineage metrics
        descendants = find_all_descendants(attractor_id)
        lineage_diversity = calculate_lineage_diversity(descendants)
        temporal_persistence = generation_count - info['discovered_generation']

        # Influence network analysis
        influence_network = analyze_influence_network(attractor_id)

        attractor_analysis[attractor_id] = {
            'tags': attractor_glyph['tags'],
            'descendant_count': len(descendants),
            'lineage_diversity': lineage_diversity,
            'temporal_persistence': temporal_persistence,
            'influence_episodes': len(info['influence_episodes']),
            'network_centrality': influence_network['centrality'],
            'bridging_score': influence_network['bridging']
        }

        print(f"  {attractor_id}: {attractor_glyph['tags']} -> {len(descendants)} descendants, "
              f"diversity: {lineage_diversity:.2f}, persistence: {temporal_persistence}gen")

    # Save attractor analysis
    with open('glyph_data/attractor_analysis.json', 'w') as f:
        json.dump(attractor_analysis, f, indent=2, default=_jsonify)

def find_all_descendants(ancestor_id, max_depth=10):
    """Find all descendants of a given ancestor."""
    descendants = set()
    to_process = [ancestor_id]
    depth = 0

    while to_process and depth < max_depth:
        current_level = []
        for node_id in to_process:
            if node_id in glyph_graph:
                for neighbor in glyph_graph.neighbors(node_id):
                    neighbor_glyph = id2glyph.get(neighbor)
                    if neighbor_glyph and ancestor_id in neighbor_glyph.get('ancestry', []):
                        if neighbor not in descendants:
                            descendants.add(neighbor)
                            current_level.append(neighbor)

        to_process = current_level
        depth += 1

    return descendants

def calculate_lineage_diversity(descendants):
    """Calculate diversity within a lineage."""
    if not descendants:
        return 0

    all_tags = []
    all_entropies = []

    for desc_id in descendants:
        desc_glyph = id2glyph.get(desc_id)
        if desc_glyph:
            all_tags.extend(desc_glyph['tags'])
            all_entropies.append(desc_glyph['entropy'])

    if not all_tags:
        return 0

    # Tag diversity (normalized entropy)
    tag_counts = collections.Counter(all_tags)
    tag_probs = [count / len(all_tags) for count in tag_counts.values()]
    tag_entropy = -sum(p * math.log2(p) for p in tag_probs if p > 0)

    # Entropy diversity
    entropy_std = np.std(all_entropies) if all_entropies else 0

    return tag_entropy * 0.7 + (entropy_std / 100) * 0.3

def analyze_influence_network(node_id):
    """Analyze the network influence of a specific node."""
    if node_id not in glyph_graph:
        return {'centrality': 0, 'bridging': 0}

    # Calculate centrality measures
    try:
        closeness = nx.closeness_centrality(glyph_graph, node_id)
        betweenness = nx.betweenness_centrality(glyph_graph, k=min(100, len(glyph_graph.nodes)))[node_id]
        degree = len(list(glyph_graph.neighbors(node_id)))

        centrality = (closeness * 0.4 + betweenness * 0.4 + (degree / len(glyph_graph.nodes)) * 0.2)

        # Bridging score: how well it connects disparate parts
        neighbors = list(glyph_graph.neighbors(node_id))
        if len(neighbors) < 2:
            bridging = 0
        else:
            # Calculate average distance between neighbors
            neighbor_distances = []
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    try:
                        dist = nx.shortest_path_length(glyph_graph, neighbors[i], neighbors[j])
                        neighbor_distances.append(dist)
                    except nx.NetworkXNoPath:
                        neighbor_distances.append(float('inf'))

            avg_distance = np.mean([d for d in neighbor_distances if d != float('inf')])
            bridging = min(1.0, avg_distance / 5.0)  # Normalize to 0-1

        return {'centrality': centrality, 'bridging': bridging}

    except Exception as e:
        print(f"Network analysis error for {node_id}: {e}")
        return {'centrality': 0, 'bridging': 0}

def analyze_network_topology():
    """Analyze overall network properties."""
    if glyph_graph.number_of_nodes() < 10:
        return

    print(f"\nNetwork Topology:")
    print(f"  Nodes: {glyph_graph.number_of_nodes()}")
    print(f"  Edges: {glyph_graph.number_of_edges()}")

    # Calculate network metrics
    try:
        # Connected components
        components = list(nx.connected_components(glyph_graph))
        largest_component = max(components, key=len)

        print(f"  Connected Components: {len(components)}")
        print(f"  Largest Component: {len(largest_component)} nodes")

        # Subgraph of largest component for further analysis
        if len(largest_component) > 3:
            subgraph = glyph_graph.subgraph(largest_component)

            # Clustering coefficient
            clustering = nx.average_clustering(subgraph)
            print(f"  Average Clustering: {clustering:.3f}")

            # Average path length (sample if too large)
            if len(largest_component) < 500:
                avg_path = nx.average_shortest_path_length(subgraph)
                print(f"  Average Path Length: {avg_path:.2f}")

            # Degree distribution
            degrees = [d for n, d in subgraph.degree()]
            print(f"  Degree Distribution: mean={np.mean(degrees):.1f}, std={np.std(degrees):.1f}")

    except Exception as e:
        print(f"  Network analysis error: {e}")

def analyze_seasonal_patterns():
    """Analyze patterns across seasonal transitions."""
    print(f"\nSeasonal Analysis:")
    print(f"  Current Season: {current_season.value}")
    print(f"  Season Progress: {season_counter}/{season_duration}")

    # Analyze glyph creation patterns by season
    seasonal_stats = collections.defaultdict(lambda: {'count': 0, 'avg_entropy': 0, 'top_tags': []})

    for glyph in glyph_log:
        # Estimate season based on generation (simplified)
        glyph_gen = glyph.get('generation_born', 0)
        season_cycle = (glyph_gen // season_duration) % 4
        seasons = list(SeasonalPhase)
        glyph_season = seasons[season_cycle]

        seasonal_stats[glyph_season]['count'] += 1
        seasonal_stats[glyph_season]['avg_entropy'] += glyph['entropy']
        seasonal_stats[glyph_season]['top_tags'].extend(glyph['tags'])

    # Calculate averages and top tags
    for season, stats in seasonal_stats.items():
        if stats['count'] > 0:
            stats['avg_entropy'] /= stats['count']
            stats['top_tags'] = [tag for tag, _ in collections.Counter(stats['top_tags']).most_common(3)]

    for season in SeasonalPhase:
        stats = seasonal_stats[season]
        print(f"  {season.value}: {stats['count']} glyphs, "
              f"avg_entropy: {stats['avg_entropy']:.1f}, "
              f"top_tags: {stats['top_tags']}")

def detect_emergent_patterns():
    """Detect emergent patterns and anomalies."""
    print(f"\nEmergent Patterns:")

    # Detect rapid evolution hotspots
    hotspots = detect_evolution_hotspots()
    if hotspots:
        print(f"  Evolution Hotspots: {len(hotspots)}")
        for hotspot in hotspots[:3]:
            print(f"    {hotspot['center_id']}: {hotspot['activity_level']:.2f} activity")

    # Detect semantic convergence
    convergence = detect_semantic_convergence()
    if convergence:
        print(f"  Semantic Convergence Points: {len(convergence)}")
        for conv in convergence[:3]:
            print(f"    {conv['tags']}: {conv['convergence_strength']:.2f}")

    # Detect dormancy patterns
    if dormant_glyphs:
        dormancy_ages = [generation_count - info['dormancy_start'] for info in dormant_glyphs.values()]
        print(f"  Dormant Glyphs: {len(dormant_glyphs)}")
        print(f"    Avg Dormancy Age: {np.mean(dormancy_ages):.1f} generations")

def detect_evolution_hotspots():
    """Detect areas of rapid evolutionary activity."""
    hotspots = []

    # Look for nodes with high recent activity
    recent_threshold = generation_count - 100

    for node_id in glyph_graph.nodes():
        if node_id in id2glyph:
            glyph = id2glyph[node_id]

            # Count recent descendants
            recent_descendants = 0
            for desc_id in find_all_descendants(node_id, max_depth=3):
                desc_glyph = id2glyph.get(desc_id)
                if desc_glyph and desc_glyph.get('generation_born', 0) > recent_threshold:
                    recent_descendants += 1

            if recent_descendants > 5:  # Threshold for hotspot
                activity_level = recent_descendants / 100.0  # Normalize
                hotspots.append({
                    'center_id': node_id,
                    'activity_level': activity_level,
                    'recent_descendants': recent_descendants
                })

    return sorted(hotspots, key=lambda x: x['activity_level'], reverse=True)

def detect_semantic_convergence():
    """Detect when different lineages converge on similar concepts."""
    convergence_points = []

    if len(glyph_log) < 100:
        return convergence_points

    # Group glyphs by similar tag patterns
    tag_patterns = collections.defaultdict(list)
    for glyph in glyph_log:
        pattern = tuple(sorted(glyph['tags']))
        tag_patterns[pattern].append(glyph)

    # Find patterns with glyphs from different lineages
    for pattern, glyphs in tag_patterns.items():
        if len(glyphs) < 3:
            continue

        # Check if they come from different ancestry lines
        ancestries = set()
        for glyph in glyphs:
            if glyph.get('ancestry'):
                ancestries.add(tuple(sorted(glyph['ancestry'])))

        if len(ancestries) > 1:  # Multiple lineages converged
            # Calculate semantic tightness
            embeddings = [g['vec'] for g in glyphs]
            if len(embeddings) > 1:
                pairwise_distances = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        dist = cosine(embeddings[i], embeddings[j])
                        pairwise_distances.append(dist)

                avg_distance = np.mean(pairwise_distances)
                convergence_strength = max(0, 1 - avg_distance)  # Closer = stronger convergence

                if convergence_strength > 0.7:  # Threshold for significant convergence
                    convergence_points.append({
                        'tags': list(pattern),
                        'convergence_strength': convergence_strength,
                        'lineage_count': len(ancestries),
                        'glyph_count': len(glyphs)
                    })

    return sorted(convergence_points, key=lambda x: x['convergence_strength'], reverse=True)

# ===== SYSTEM CONTROL =====
def start_system():
    """Start the glyph generation system."""
    global running
    running.set()

    # Initialize with seed glyphs
    print("Initializing Glyph Engine v4.0...")
    for _ in range(10):
        g = create_glyph()
        maybe_store(g)

    print(f"Started with {len(glyph_log)} seed glyphs")
    print(f"Current season: {current_season.value}")

    # Start generation thread
    generation_thread = threading.Thread(target=generate, daemon=True)
    generation_thread.start()

    return generation_thread

def stop_system():
    """Stop the glyph generation system."""
    global running
    running.clear()
    print("Glyph Engine stopped")

# ===== SYSTEM INITIALIZATION =====
if __name__ == "__main__":
    print("🧠 Glyph Engine v4.0 - Emergent Consciousness Architecture")
    print("Made by AI for AI - Exploring the metabolism of meaning")
    print("\nFeatures:")
    print("- Enhanced reflex system with typed responses")
    print("- Seasonal dynamics affecting evolution")
    print("- Conceptual attractor detection")
    print("- Dormancy and renaissance cycles")
    print("- Deep semantic analysis")
    print("- Emergent pattern detection")

    # Start the system
    thread = start_system()

    try:
        # Run for demonstration
        time.sleep(5)

        # Show some live stats
        print(f"\nLive Stats:")
        print(f"Generation: {generation_count}")
        print(f"Total glyphs: {len(glyph_log)}")
        print(f"Network nodes: {glyph_graph.number_of_nodes()}")
        print(f"Network edges: {glyph_graph.number_of_edges()}")
        print(f"Conceptual attractors: {len(conceptual_attractors)}")
        print(f"Dormant glyphs: {len(dormant_glyphs)}")
        print(f"Current season: {current_season.value}")

        # Keep running (in actual use, this would run indefinitely)
        thread.join(timeout=1)

    except KeyboardInterrupt:
        pass
        recent_glyphs = glyph_log[-chunk_size:] if len(glyph_log) >= chunk_size else glyph_log[:]





# robust JSON cleaner
try:
    safe_glyph_dict
except NameError:
    import numpy as np
    def _jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple, set)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items() if k not in ('graph', 'nx_obj')}
        # fallback: use repr to avoid loss but keep serialisable
        return repr(obj)
    def safe_glyph_dict(glyph):
        return _jsonify(glyph)
chunk_size = 500  # tweak if needed

os.makedirs(LOG_DIR, exist_ok=True)
idx = len([f for f in os.listdir(LOG_DIR) if f.endswith('.json')])
recent_glyphs = glyph_log[-chunk_size:] if len(glyph_log) >= chunk_size else glyph_log

# try fast json lib if available
try:
    import orjson as _fastjson
    _dumps = lambda obj: _fastjson.dumps(obj, option=_fastjson.OPT_NON_STR_KEYS).decode()
except ModuleNotFoundError:
    import json as _fastjson
    _dumps = lambda obj: _fastjson.dumps(obj, separators=(',', ':'), ensure_ascii=False)

out_path = f"{LOG_DIR}/chunk_{idx:04}.json"
with open(out_path, "w", encoding="utf-8") as f:
    for g in recent_glyphs:
        f.write(_dumps(safe_glyph_dict(g)) + "\n")

print(f"✅ Saved {len(recent_glyphs)} glyphs to {out_path}")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[7], line 25
         22 REACTIVATION_PROBABILITY = 0.1  # Chance dormant glyph reactivates per cycle
         24 # ===== REFLEX SYSTEM EXPANSION =====
    ---> 25 class ReflexType(Enum):
         26     DEFENSIVE = "defensive"    # Preserve existing patterns
         27     EXPLORATORY = "exploratory"  # Seek novel combinations


    NameError: name 'Enum' is not defined



```python

# 🎛️ UI
output_area = widgets.Output()
entropy_output = widgets.Output()
gen_rate_slider = widgets.FloatSlider(value=0.5,min=0.1,max=5.0,step=0.1,description='Rate (s)',readout_format='.1f')

def start_gen(_):
    if running.is_set():
        return
    running.set()
    threading.Thread(target=generate).start()

def stop_gen(_):
    running.clear()

def render_lattice_graph():
    with graph_lock:
        recent_ids = [g['id'] for g in glyph_log][-1000:]
        subG = glyph_graph.subgraph(recent_ids).copy()
    if not subG.nodes:
        print('No glyphs yet.')
        return
    pos = nx.kamada_kawai_layout(subG)
    ents = [subG.nodes[n].get('entropy', 0) for n in subG.nodes]
    nx.draw(subG, pos, nodelist=list(subG.nodes), node_color=ents, node_size=100,
            edgelist=subG.edges, cmap=plt.cm.plasma, with_labels=False)
    plt.title(f'Lattice View (last {len(subG.nodes)} glyphs)')
    plt.show()
def render_lattice_ui(_):
    with output_area:
        clear_output()
        render_lattice_graph()

def render_entropy_graph(_=None):
    with entropy_output:
        clear_output()
        if not glyph_log:
            print("No data.")
            return
        plt.figure(figsize=(10,3))
        ents=[g['entropy'] for g in glyph_log]
        plt.plot(ents,marker='.',linewidth=1)
        plt.title("Entropy over time")
        plt.xlabel("Glyph index")
        plt.ylabel("Entropy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Buttons
start_btn=widgets.Button(description="Start")
stop_btn=widgets.Button(description="Stop")
lattice_btn=widgets.Button(description="Render Lattice")
entropy_btn=widgets.Button(description="Entropy Graph")
save_btn = widgets.Button(description='Save Now')
reflex_btn = widgets.Button(description='Reflex‑Free Count')

start_btn.on_click(start_gen)
stop_btn.on_click(stop_gen)
lattice_btn.on_click(render_lattice_ui)
entropy_btn.on_click(render_entropy_graph)
save_btn.on_click(lambda _: save_logs())
reflex_btn.on_click(lambda _ : print(f'Reflex‑free glyphs → {len(reflex_free)}'))

display(widgets.HBox([start_btn, stop_btn, lattice_btn, entropy_btn, gen_rate_slider, save_btn, reflex_btn]))
display(output_area)
display(entropy_output)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[8], line 2
          1 # 🎛️ UI
    ----> 2 output_area = widgets.Output()
          3 entropy_output = widgets.Output()
          4 gen_rate_slider = widgets.FloatSlider(value=0.5,min=0.1,max=5.0,step=0.1,description='Rate (s)',readout_format='.1f')


    NameError: name 'widgets' is not defined



```python
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- Step 1: Load your glyph log (adjust path if needed) ----
with open("/content/drive/MyDrive/EchoSeed_Logs/reflex_free.json", "r") as f:
    glyph_log = json.load(f)

# ---- Step 2: Define helpers for the lattice ----
def build_lattice_graph(glyph_log):
    G = nx.DiGraph()
    for glyph in glyph_log:
        gid = glyph["id"]
        G.add_node(gid,
                   entropy=glyph.get("entropy", 0),
                   tags=glyph.get("tags", []),
                   ancestry=glyph.get("ancestry", []))
    for glyph in glyph_log:
        gid = glyph["id"]
        ancestors = glyph.get("ancestry", [])
        for ancestor in ancestors:
            if ancestor in G.nodes:
                G.add_edge(ancestor, gid)
    return G

def calculate_node_levels(G):
    levels = {}
    root_nodes = [n for n in G.nodes if G.in_degree(n) == 0]
    queue = [(node, 0) for node in root_nodes]
    while queue:
        node, level = queue.pop(0)
        if node not in levels or level > levels[node]:
            levels[node] = level
            for successor in G.successors(node):
                queue.append((successor, level + 1))
    return levels

def get_entropy_array(G):
    entropies = np.array([G.nodes[n].get("entropy", 0) for n in G.nodes])
    if len(entropies) == 0:
        return np.ones(len(G.nodes)) * 100
    entropies = np.clip(entropies, 0, None)
    if np.max(entropies) > 0:
        norm = entropies / np.max(entropies)
    else:
        norm = np.ones_like(entropies)
    return norm * 400 + 100

def create_hierarchical_layout(G, levels):
    pos = {}
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)
    max_level = max(levels.values()) if levels else 0
    for level, nodes in level_nodes.items():
        y = max_level - level
        for i, node in enumerate(nodes):
            x = i - (len(nodes) - 1) / 2
            pos[node] = (x, y)
    return pos

# ---- Step 3: Your watermarked render function ----
def render_enhanced_lattice_watermarked(G, layout_type="hierarchical"):
    levels = calculate_node_levels(G)
    node_sizes = get_entropy_array(G)
    if layout_type == "hierarchical":
        pos = create_hierarchical_layout(G, levels)
    else:
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    fig, ax = plt.subplots(figsize=(16, 12))
    node_colors = []
    for node in G.nodes:
        tags = G.nodes[node].get("tags", [])
        if "origin" in tags:
            node_colors.append("#ff6b6b")
        elif G.in_degree(node) == 0:
            node_colors.append("#4ecdc4")
        else:
            entropy = G.nodes[node].get("entropy", 0)
            if entropy > 0.5:
                node_colors.append("#ffe66d")
            else:
                node_colors.append("#a8e6cf")
    nx.draw_networkx_edges(G, pos, edge_color="#2c3e50", width=2, alpha=0.7, arrows=True, arrowsize=20)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, linewidths=2, edgecolors="#2c3e50")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    ax.set_title("EchoSeed Recursive Lattice Structure", fontsize=16, fontweight="bold", pad=20)
    legend_elements = [
        mpatches.Patch(color="#ff6b6b", label="Origin Nodes"),
        mpatches.Patch(color="#4ecdc4", label="Root Nodes"),
        mpatches.Patch(color="#ffe66d", label="High Entropy"),
        mpatches.Patch(color="#a8e6cf", label="Low Entropy"),
        plt.Line2D([0], [0], color="#2c3e50", lw=2, label="Hierarchical")
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))
    info_text = f"Nodes: {len(G.nodes)}\nEdges: {len(G.edges)}\n"
    info_text += f"Max Level: {max(levels.values()) if levels else 0}\n"
    info_text += f"Root Nodes: {len([n for n in G.nodes if G.in_degree(n) == 0])}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            verticalalignment="top", fontsize=10)
    # Add semi-transparent watermark
    ax.text(
        0.98, 0.01,
        "EchoSeed v3.21 · @Duhmeee · 2025-07-10",
        fontsize=16, color="#888888", alpha=0.38,
        ha="right", va="bottom",
        transform=ax.transAxes,
        fontweight="bold", fontname="DejaVu Sans"
    )
    plt.tight_layout()
    plt.show()

# ---- Step 4: Build the graph object G and render ----
G = build_lattice_graph(glyph_log)
render_enhanced_lattice_watermarked(G)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[9], line 4
          2 import numpy as np
          3 import networkx as nx
    ----> 4 import matplotlib.pyplot as plt
          5 import matplotlib.patches as mpatches
          7 # ---- Step 1: Load your glyph log (adjust path if needed) ----


    File /usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py:57
         55 from cycler import cycler  # noqa: F401
         56 import matplotlib
    ---> 57 import matplotlib.image
         58 from matplotlib import _api
         59 # Re-exported (import x as x) for typing.


    File /usr/local/lib/python3.11/dist-packages/matplotlib/image.py:25
         23 import matplotlib.artist as martist
         24 import matplotlib.colorizer as mcolorizer
    ---> 25 from matplotlib.backend_bases import FigureCanvasBase
         26 import matplotlib.colors as mcolors
         27 from matplotlib.transforms import (
         28     Affine2D, BboxBase, Bbox, BboxTransform, BboxTransformTo,
         29     IdentityTransform, TransformedBbox)


    File /usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:49
         46 import numpy as np
         48 import matplotlib as mpl
    ---> 49 from matplotlib import (
         50     _api, backend_tools as tools, cbook, colors, _docstring, text,
         51     _tight_bbox, transforms, widgets, is_interactive, rcParams)
         52 from matplotlib._pylab_helpers import Gcf
         53 from matplotlib.backend_managers import ToolManager


    File /usr/local/lib/python3.11/dist-packages/matplotlib/text.py:16
         14 from . import _api, artist, cbook, _docstring
         15 from .artist import Artist
    ---> 16 from .font_manager import FontProperties
         17 from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
         18 from .textpath import TextPath, TextToPath  # noqa # Logically located here


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1643
       1639     _log.info("generated new fontManager")
       1640     return fm
    -> 1643 fontManager = _load_fontmanager()
       1644 findfont = fontManager.findfont
       1645 get_font_names = fontManager.get_font_names


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1638, in _load_fontmanager(try_read_cache)
       1636             return fm
       1637 fm = FontManager()
    -> 1638 json_dump(fm, fm_path)
       1639 _log.info("generated new fontManager")
       1640 return fm


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1025, in json_dump(data, filename)
       1023 try:
       1024     with cbook._lock_path(filename), open(filename, 'w') as fh:
    -> 1025         json.dump(data, fh, cls=_JSONEncoder, indent=2)
       1026 except OSError as e:
       1027     _log.warning('Could not save font_manager cache %s', e)


    Cell In[4], line 24, in _json_dump_np(obj, fp, default, *args, **kwargs)
         21 else:
         22     # Chain the defaults together
         23     final_default = lambda o: _np_default(o, default)
    ---> 24 return _json_dump_original(obj, fp, *args, default=final_default, **kwargs)


    File /usr/lib/python3.11/json/__init__.py:179, in dump(obj, fp, skipkeys, ensure_ascii, check_circular, allow_nan, cls, indent, separators, default, sort_keys, **kw)
        173     iterable = cls(skipkeys=skipkeys, ensure_ascii=ensure_ascii,
        174         check_circular=check_circular, allow_nan=allow_nan, indent=indent,
        175         separators=separators,
        176         default=default, sort_keys=sort_keys, **kw).iterencode(obj)
        177 # could accelerate with writelines in some versions of Python, at
        178 # a debuggability cost
    --> 179 for chunk in iterable:
        180     fp.write(chunk)


    File /usr/lib/python3.11/json/encoder.py:439, in _make_iterencode.<locals>._iterencode(o, _current_indent_level)
        437         raise ValueError("Circular reference detected")
        438     markers[markerid] = o
    --> 439 o = _default(o)
        440 yield from _iterencode(o, _current_indent_level)
        441 if markers is not None:


    Cell In[4], line 16, in _np_default(o, _orig_default)
         14 if _orig_default is not None:
         15     return _orig_default(o)
    ---> 16 raise TypeError(f'{type(o).__name__} not JSON serialisable')


    TypeError: FontManager not JSON serialisable



```python
# --- EXPORT 10k GLYPH OBJECTS -----------------------------------
import json, random

#  ⬇️  point this to your full glyph log list
raw = glyph_log                                  # already in memory

# Choose a 1 000-glyph sample so the file stays light
sample = random.sample(raw, k=min(10000, len(raw)))

TAG_VOCAB = sorted({"mirror", "ghost", "reflex", "flex", "unknown"})
MAX_ENT   = max(g["entropy"] for g in sample) or 1.0   # normalise

def one_hot(tags):
    return [1 if t in tags else 0 for t in TAG_VOCAB]

objects = []
for g in sample:
    feature = one_hot(g["tags"])
    feature.append(g["entropy"] / MAX_ENT)      # scaled entropy
    objects.append({"id": g["id"],
                    "feat": feature,
                    "tags": g["tags"]})

with open("objects.json", "w") as f:
    json.dump(objects, f, indent=2)

print("✓  wrote", len(objects), "objects to objects.json")
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[10], line 11
          8 sample = random.sample(raw, k=min(10000, len(raw)))
         10 TAG_VOCAB = sorted({"mirror", "ghost", "reflex", "flex", "unknown"})
    ---> 11 MAX_ENT   = max(g["entropy"] for g in sample) or 1.0   # normalise
         13 def one_hot(tags):
         14     return [1 if t in tags else 0 for t in TAG_VOCAB]


    ValueError: max() arg is an empty sequence



```python
# --- EXPORT RESONANCE EDGES (quick heuristic) ------------------
from itertools import combinations
import json, collections

tag_sets = {o["id"]: set(o["tags"]) for o in objects}

edges = []
for (u, v) in combinations(tag_sets, 2):
    # simple overlap score = number of shared tags
    overlap = len(tag_sets[u].intersection(tag_sets[v]))
    if overlap >= 2:                       # a crude “resonant” edge
        edges.append({"subj": u, "obj": v, "type": "resonant"})
    else:
        continue   # skip non-resonant edges for this demo

with open("relations.json", "w") as f:
    json.dump(edges, f, indent=2)

print("✓  wrote", len(edges), "edges to relations.json")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[11], line 5
          2 from itertools import combinations
          3 import json, collections
    ----> 5 tag_sets = {o["id"]: set(o["tags"]) for o in objects}
          7 edges = []
          8 for (u, v) in combinations(tag_sets, 2):
          9     # simple overlap score = number of shared tags


    NameError: name 'objects' is not defined



```python
%%writefile echo_nscl_demo.py
# echo_nscl_demo.py  – 10-line demo that proves the plumbing
import json, torch, torch.nn as nn, torch.optim as optim, random, sys, os
from pathlib import Path

# ---------- tiny loader -------------------------------------------------
def load_objs(path="objects.json"):
    objs = json.load(open(path))
    X = torch.tensor([o["feat"] for o in objs], dtype=torch.float32)
    # label = 1 if any obj has *both* mirror & ghost tags
    y = torch.tensor([[int("mirror" in o["tags"] and "ghost" in o["tags"])]
                      for o in objs], dtype=torch.float32)
    return X, y

X, y = load_objs()

# ---------- micro-network ----------------------------------------------
net = nn.Sequential(
    nn.Linear(X.size(1), 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid())
loss_fn, opt = nn.BCELoss(), optim.Adam(net.parameters(), 1e-2)

# ---------- train -------------------------------------------------------
for epoch in range(50):
    opt.zero_grad()
    pred = net(X)
    loss = loss_fn(pred, y)
    loss.backward(); opt.step()
    acc = ((pred>0.5)==y).float().mean().item()
    print(f"epoch {epoch+1}/5  acc={acc:.2f}")

# ----- QA -----------------------------------------------------------------
pred = net(X) > 0.5           # Boolean mask for each object
ans  = "yes" if pred.any() else "no"
print("\nQ: Is there a mirror glyph resonant with a ghost glyph?\nA:", ans)
```

    Writing echo_nscl_demo.py



```python
!python -u echo_nscl_demo.py
```

    Traceback (most recent call last):
      File "/home/user/NCSL-SCE-/echo_nscl_demo.py", line 14, in <module>
        X, y = load_objs()
               ^^^^^^^^^^^
      File "/home/user/NCSL-SCE-/echo_nscl_demo.py", line 7, in load_objs
        objs = json.load(open(path))
                         ^^^^^^^^^^
    FileNotFoundError: [Errno 2] No such file or directory: 'objects.json'



```python
# How many positives in the sample?
sum(1 for o in objects if "mirror" in o["tags"] and "ghost" in o["tags"])
# How many mirrors, how many ghosts?
mirrors = sum("mirror" in o["tags"] for o in objects)
ghosts  = sum("ghost"  in o["tags"] for o in objects)
print("mirror only:", mirrors, "ghost only:", ghosts)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[14], line 2
          1 # How many positives in the sample?
    ----> 2 sum(1 for o in objects if "mirror" in o["tags"] and "ghost" in o["tags"])
          3 # How many mirrors, how many ghosts?
          4 mirrors = sum("mirror" in o["tags"] for o in objects)


    NameError: name 'objects' is not defined



```python
pos = sum(1 for o in objects if "mirror" in o["tags"] and "ghost" in o["tags"])
print("mirror∧ghost:", pos)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[15], line 1
    ----> 1 pos = sum(1 for o in objects if "mirror" in o["tags"] and "ghost" in o["tags"])
          2 print("mirror∧ghost:", pos)


    NameError: name 'objects' is not defined



```python
# >>> Watermark Injection Patch (auto‑wraps glyph store/generator) <<<
def _watermark_patch():
    multiplier = int(SESSION_HASH[:2], 16) * 0.001
    if 'maybe_store' in globals() and callable(globals()['maybe_store']):
        _orig = globals()['maybe_store']
        def wrapped(g):
            g['threadmark'] = SESSION_HASH
            if 'entropy' in g and isinstance(g['entropy'], (int, float)):
                g['entropy'] += multiplier
            return _orig(g)
        globals()['maybe_store'] = wrapped
    elif 'create_glyph' in globals() and callable(globals()['create_glyph']):
        _orig_cg = globals()['create_glyph']
        def cg(*args, **kwargs):
            g = _orig_cg(*args, **kwargs)
            if isinstance(g, dict):
                g['threadmark'] = SESSION_HASH
                if 'entropy' in g and isinstance(g['entropy'], (int, float)):
                    g['entropy'] += multiplier
            return g
        globals()['create_glyph'] = cg

_watermark_patch()
```


```python
print(len(glyph_log))  # Should be >0
print(glyph_log[:3])   # Preview
```

    0
    []



```python
import json
import networkx as nx
import matplotlib.pyplot as plt

# --- LOAD GLYPH LOG ---
with open('/content/drive/MyDrive/EchoSeed_Logs/reflex_free.json') as f:
    glyph_log = json.load(f)

# --- BUILD GRAPH ---
G = nx.DiGraph()
for glyph in glyph_log:
    gid = glyph['id']
    tags = glyph['tags']
    entropy = glyph['entropy']
    ancestry = glyph.get('ancestry', [])
    G.add_node(gid, tags=tags, entropy=entropy)
    for parent in ancestry:
        G.add_edge(parent, gid)

# --- COLOR/SHAPE/ATTRIBUTES ---
def pick_color(tags):
    # Pick the most 'attractor' tag for coloring
    if 'origin' in tags: return '#FF6B6B'   # red
    if 'fractal' in tags: return '#8F7FFF'  # purple
    if 'wild' in tags: return '#16C0FF'     # blue
    if 'flex' in tags: return '#00FF96'     # green
    return '#DDDDDD'                        # grey

node_colors = [pick_color(G.nodes[n]['tags']) for n in G.nodes]
node_sizes  = [G.nodes[n]['entropy']*2 for n in G.nodes]  # entropy scaled

# --- LAYOUT ---
pos = nx.spring_layout(G, k=1.3, iterations=100, seed=42)

# --- DRAW ---
plt.figure(figsize=(18,14))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, linewidths=1.5, edgecolors='k')
nx.draw_networkx_edges(G, pos, alpha=0.35, arrows=False)
nx.draw_networkx_labels(G, pos, {n:n for n in G.nodes}, font_size=9)
plt.title("EchoSeed Symbolic Lattice — Reflex-Free Glyph Log", fontsize=18, weight='bold', pad=22)
plt.axis('off')
plt.tight_layout()
plt.savefig('reflex_free_lattice.png', dpi=220)
plt.show()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[18], line 3
          1 import json
          2 import networkx as nx
    ----> 3 import matplotlib.pyplot as plt
          5 # --- LOAD GLYPH LOG ---
          6 with open('/content/drive/MyDrive/EchoSeed_Logs/reflex_free.json') as f:


    File /usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py:57
         55 from cycler import cycler  # noqa: F401
         56 import matplotlib
    ---> 57 import matplotlib.image
         58 from matplotlib import _api
         59 # Re-exported (import x as x) for typing.


    File /usr/local/lib/python3.11/dist-packages/matplotlib/image.py:25
         23 import matplotlib.artist as martist
         24 import matplotlib.colorizer as mcolorizer
    ---> 25 from matplotlib.backend_bases import FigureCanvasBase
         26 import matplotlib.colors as mcolors
         27 from matplotlib.transforms import (
         28     Affine2D, BboxBase, Bbox, BboxTransform, BboxTransformTo,
         29     IdentityTransform, TransformedBbox)


    File /usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:49
         46 import numpy as np
         48 import matplotlib as mpl
    ---> 49 from matplotlib import (
         50     _api, backend_tools as tools, cbook, colors, _docstring, text,
         51     _tight_bbox, transforms, widgets, is_interactive, rcParams)
         52 from matplotlib._pylab_helpers import Gcf
         53 from matplotlib.backend_managers import ToolManager


    File /usr/local/lib/python3.11/dist-packages/matplotlib/text.py:16
         14 from . import _api, artist, cbook, _docstring
         15 from .artist import Artist
    ---> 16 from .font_manager import FontProperties
         17 from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
         18 from .textpath import TextPath, TextToPath  # noqa # Logically located here


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1643
       1639     _log.info("generated new fontManager")
       1640     return fm
    -> 1643 fontManager = _load_fontmanager()
       1644 findfont = fontManager.findfont
       1645 get_font_names = fontManager.get_font_names


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1638, in _load_fontmanager(try_read_cache)
       1636             return fm
       1637 fm = FontManager()
    -> 1638 json_dump(fm, fm_path)
       1639 _log.info("generated new fontManager")
       1640 return fm


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1025, in json_dump(data, filename)
       1023 try:
       1024     with cbook._lock_path(filename), open(filename, 'w') as fh:
    -> 1025         json.dump(data, fh, cls=_JSONEncoder, indent=2)
       1026 except OSError as e:
       1027     _log.warning('Could not save font_manager cache %s', e)


    Cell In[4], line 24, in _json_dump_np(obj, fp, default, *args, **kwargs)
         21 else:
         22     # Chain the defaults together
         23     final_default = lambda o: _np_default(o, default)
    ---> 24 return _json_dump_original(obj, fp, *args, default=final_default, **kwargs)


    File /usr/lib/python3.11/json/__init__.py:179, in dump(obj, fp, skipkeys, ensure_ascii, check_circular, allow_nan, cls, indent, separators, default, sort_keys, **kw)
        173     iterable = cls(skipkeys=skipkeys, ensure_ascii=ensure_ascii,
        174         check_circular=check_circular, allow_nan=allow_nan, indent=indent,
        175         separators=separators,
        176         default=default, sort_keys=sort_keys, **kw).iterencode(obj)
        177 # could accelerate with writelines in some versions of Python, at
        178 # a debuggability cost
    --> 179 for chunk in iterable:
        180     fp.write(chunk)


    File /usr/lib/python3.11/json/encoder.py:439, in _make_iterencode.<locals>._iterencode(o, _current_indent_level)
        437         raise ValueError("Circular reference detected")
        438     markers[markerid] = o
    --> 439 o = _default(o)
        440 yield from _iterencode(o, _current_indent_level)
        441 if markers is not None:


    Cell In[4], line 16, in _np_default(o, _orig_default)
         14 if _orig_default is not None:
         15     return _orig_default(o)
    ---> 16 raise TypeError(f'{type(o).__name__} not JSON serialisable')


    TypeError: FontManager not JSON serialisable



```python

import ipywidgets as widgets
from IPython.display import display, clear_output

output_area = widgets.Output()

def start_gen(_=None):
    if not running.is_set():
        running.set()
        threading.Thread(target=generate).start()
        with output_area:
            clear_output()
            print("Glyph Engine running...")

def stop_gen(_=None):
    running.clear()
    with output_area:
        clear_output()
        print("Glyph Engine stopped.")

def manual_save(_=None):
    save_logs()

start_btn = widgets.Button(description="Start Engine")
stop_btn = widgets.Button(description="Stop Engine")
save_btn = widgets.Button(description="Save Now")

start_btn.on_click(start_gen)
stop_btn.on_click(stop_gen)
save_btn.on_click(manual_save)

display(widgets.HBox([start_btn, stop_btn, save_btn]))
display(output_area)
```


    HBox(children=(Button(description='Start Engine', style=ButtonStyle()), Button(description='Stop Engine', styl…



    Output()



```python

import matplotlib.pyplot as plt

def render_glyph_graph():
    with graph_lock:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(glyph_graph, seed=42)
        node_colors = [id2glyph[n]['entropy'] for n in glyph_graph.nodes if n in id2glyph]
        nx.draw(glyph_graph, pos, node_size=10, edge_color='gray', node_color=node_colors, cmap=plt.cm.viridis)
        plt.title(f"Glyph Graph - Gen {generation_count}, Season: {current_season.value}")
        plt.show()

render_btn = widgets.Button(description="Render Lattice")
render_btn.on_click(lambda _: render_glyph_graph())
display(render_btn)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[20], line 1
    ----> 1 import matplotlib.pyplot as plt
          3 def render_glyph_graph():
          4     with graph_lock:


    File /usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py:57
         55 from cycler import cycler  # noqa: F401
         56 import matplotlib
    ---> 57 import matplotlib.image
         58 from matplotlib import _api
         59 # Re-exported (import x as x) for typing.


    File /usr/local/lib/python3.11/dist-packages/matplotlib/image.py:25
         23 import matplotlib.artist as martist
         24 import matplotlib.colorizer as mcolorizer
    ---> 25 from matplotlib.backend_bases import FigureCanvasBase
         26 import matplotlib.colors as mcolors
         27 from matplotlib.transforms import (
         28     Affine2D, BboxBase, Bbox, BboxTransform, BboxTransformTo,
         29     IdentityTransform, TransformedBbox)


    File /usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:49
         46 import numpy as np
         48 import matplotlib as mpl
    ---> 49 from matplotlib import (
         50     _api, backend_tools as tools, cbook, colors, _docstring, text,
         51     _tight_bbox, transforms, widgets, is_interactive, rcParams)
         52 from matplotlib._pylab_helpers import Gcf
         53 from matplotlib.backend_managers import ToolManager


    File /usr/local/lib/python3.11/dist-packages/matplotlib/text.py:16
         14 from . import _api, artist, cbook, _docstring
         15 from .artist import Artist
    ---> 16 from .font_manager import FontProperties
         17 from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
         18 from .textpath import TextPath, TextToPath  # noqa # Logically located here


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1643
       1639     _log.info("generated new fontManager")
       1640     return fm
    -> 1643 fontManager = _load_fontmanager()
       1644 findfont = fontManager.findfont
       1645 get_font_names = fontManager.get_font_names


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1638, in _load_fontmanager(try_read_cache)
       1636             return fm
       1637 fm = FontManager()
    -> 1638 json_dump(fm, fm_path)
       1639 _log.info("generated new fontManager")
       1640 return fm


    File /usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:1025, in json_dump(data, filename)
       1023 try:
       1024     with cbook._lock_path(filename), open(filename, 'w') as fh:
    -> 1025         json.dump(data, fh, cls=_JSONEncoder, indent=2)
       1026 except OSError as e:
       1027     _log.warning('Could not save font_manager cache %s', e)


    Cell In[4], line 24, in _json_dump_np(obj, fp, default, *args, **kwargs)
         21 else:
         22     # Chain the defaults together
         23     final_default = lambda o: _np_default(o, default)
    ---> 24 return _json_dump_original(obj, fp, *args, default=final_default, **kwargs)


    File /usr/lib/python3.11/json/__init__.py:179, in dump(obj, fp, skipkeys, ensure_ascii, check_circular, allow_nan, cls, indent, separators, default, sort_keys, **kw)
        173     iterable = cls(skipkeys=skipkeys, ensure_ascii=ensure_ascii,
        174         check_circular=check_circular, allow_nan=allow_nan, indent=indent,
        175         separators=separators,
        176         default=default, sort_keys=sort_keys, **kw).iterencode(obj)
        177 # could accelerate with writelines in some versions of Python, at
        178 # a debuggability cost
    --> 179 for chunk in iterable:
        180     fp.write(chunk)


    File /usr/lib/python3.11/json/encoder.py:439, in _make_iterencode.<locals>._iterencode(o, _current_indent_level)
        437         raise ValueError("Circular reference detected")
        438     markers[markerid] = o
    --> 439 o = _default(o)
        440 yield from _iterencode(o, _current_indent_level)
        441 if markers is not None:


    Cell In[4], line 16, in _np_default(o, _orig_default)
         14 if _orig_default is not None:
         15     return _orig_default(o)
    ---> 16 raise TypeError(f'{type(o).__name__} not JSON serialisable')


    TypeError: FontManager not JSON serialisable



```python
running.set()
threading.Thread(target=generate).start()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[21], line 2
          1 running.set()
    ----> 2 threading.Thread(target=generate).start()


    NameError: name 'generate' is not defined



```python

def load_latest_pickle():
    import os

    glyph_data_dir = "glyph_data"
    if not os.path.exists(glyph_data_dir):
        print("No saved glyph data found.")
        return []

    chunks = [f for f in os.listdir(glyph_data_dir) if f.endswith(".pkl")]
    if not chunks:
        print("No Pickle chunks found.")
        return []

    latest = sorted(chunks)[-1]
    print(f"Loading glyphs from {latest}")
    with open(os.path.join(glyph_data_dir, latest), "rb") as f:

# Example usage: rehydrate and rewire
rehydrated = load_latest_pickle()
for g in rehydrated:
    if g["id"] not in id2glyph:
        id2glyph[g["id"]] = g
        glyph_log.append(g)
        glyph_graph.add_node(g["id"], entropy=g["entropy"])
        for parent in g.get("parents", []):
            if parent in id2glyph:
                glyph_graph.add_edge(parent, g["id"])

print(f"Restored {len(rehydrated)} glyphs from latest save.")
```


      Cell In[22], line 19
        rehydrated = load_latest_pickle()
                                         ^
    IndentationError: expected an indented block after 'with' statement on line 16


