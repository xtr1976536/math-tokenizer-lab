# Hard AST-BPE 350-ID vocabulary

Frozen model: `experiments/downstream/Wave_equation/tokenizers/ast_bpe_350.model.json`.
IDs 0-2 are control tokens; content token ID = compact vocabulary index + 3.
Escapes use Python string notation; e.g. `\\` represents one backslash byte.

| ID | Token bytes | Kind |
|---:|---|---|
| 0 | `<PAD>` | control |
| 1 | `<EOS>` | control |
| 2 | `<BOS>` | control |
| 3 | <code>&#x27;\t&#x27;</code> | base byte |
| 4 | <code>&#x27;\n&#x27;</code> | base byte |
| 5 | <code>&#x27;\r&#x27;</code> | base byte |
| 6 | <code>&#x27; &#x27;</code> | base byte |
| 7 | <code>&#x27;!&#x27;</code> | base byte |
| 8 | <code>&#x27;&quot;&#x27;</code> | base byte |
| 9 | <code>&#x27;#&#x27;</code> | base byte |
| 10 | <code>&#x27;$&#x27;</code> | base byte |
| 11 | <code>&#x27;%&#x27;</code> | base byte |
| 12 | <code>&#x27;&amp;&#x27;</code> | base byte |
| 13 | <code>&quot;&#x27;&quot;</code> | base byte |
| 14 | <code>&#x27;(&#x27;</code> | base byte |
| 15 | <code>&#x27;)&#x27;</code> | base byte |
| 16 | <code>&#x27;*&#x27;</code> | base byte |
| 17 | <code>&#x27;+&#x27;</code> | base byte |
| 18 | <code>&#x27;,&#x27;</code> | base byte |
| 19 | <code>&#x27;-&#x27;</code> | base byte |
| 20 | <code>&#x27;.&#x27;</code> | base byte |
| 21 | <code>&#x27;/&#x27;</code> | base byte |
| 22 | <code>&#x27;0&#x27;</code> | base byte |
| 23 | <code>&#x27;1&#x27;</code> | base byte |
| 24 | <code>&#x27;2&#x27;</code> | base byte |
| 25 | <code>&#x27;3&#x27;</code> | base byte |
| 26 | <code>&#x27;4&#x27;</code> | base byte |
| 27 | <code>&#x27;5&#x27;</code> | base byte |
| 28 | <code>&#x27;6&#x27;</code> | base byte |
| 29 | <code>&#x27;7&#x27;</code> | base byte |
| 30 | <code>&#x27;8&#x27;</code> | base byte |
| 31 | <code>&#x27;9&#x27;</code> | base byte |
| 32 | <code>&#x27;:&#x27;</code> | base byte |
| 33 | <code>&#x27;;&#x27;</code> | base byte |
| 34 | <code>&#x27;&lt;&#x27;</code> | base byte |
| 35 | <code>&#x27;=&#x27;</code> | base byte |
| 36 | <code>&#x27;&gt;&#x27;</code> | base byte |
| 37 | <code>&#x27;?&#x27;</code> | base byte |
| 38 | <code>&#x27;@&#x27;</code> | base byte |
| 39 | <code>&#x27;A&#x27;</code> | base byte |
| 40 | <code>&#x27;B&#x27;</code> | base byte |
| 41 | <code>&#x27;C&#x27;</code> | base byte |
| 42 | <code>&#x27;D&#x27;</code> | base byte |
| 43 | <code>&#x27;E&#x27;</code> | base byte |
| 44 | <code>&#x27;F&#x27;</code> | base byte |
| 45 | <code>&#x27;G&#x27;</code> | base byte |
| 46 | <code>&#x27;H&#x27;</code> | base byte |
| 47 | <code>&#x27;I&#x27;</code> | base byte |
| 48 | <code>&#x27;J&#x27;</code> | base byte |
| 49 | <code>&#x27;K&#x27;</code> | base byte |
| 50 | <code>&#x27;L&#x27;</code> | base byte |
| 51 | <code>&#x27;M&#x27;</code> | base byte |
| 52 | <code>&#x27;N&#x27;</code> | base byte |
| 53 | <code>&#x27;O&#x27;</code> | base byte |
| 54 | <code>&#x27;P&#x27;</code> | base byte |
| 55 | <code>&#x27;Q&#x27;</code> | base byte |
| 56 | <code>&#x27;R&#x27;</code> | base byte |
| 57 | <code>&#x27;S&#x27;</code> | base byte |
| 58 | <code>&#x27;T&#x27;</code> | base byte |
| 59 | <code>&#x27;U&#x27;</code> | base byte |
| 60 | <code>&#x27;V&#x27;</code> | base byte |
| 61 | <code>&#x27;W&#x27;</code> | base byte |
| 62 | <code>&#x27;X&#x27;</code> | base byte |
| 63 | <code>&#x27;Y&#x27;</code> | base byte |
| 64 | <code>&#x27;Z&#x27;</code> | base byte |
| 65 | <code>&#x27;[&#x27;</code> | base byte |
| 66 | <code>&#x27;\\&#x27;</code> | base byte |
| 67 | <code>&#x27;]&#x27;</code> | base byte |
| 68 | <code>&#x27;^&#x27;</code> | base byte |
| 69 | <code>&#x27;_&#x27;</code> | base byte |
| 70 | <code>&#x27;`&#x27;</code> | base byte |
| 71 | <code>&#x27;a&#x27;</code> | base byte |
| 72 | <code>&#x27;b&#x27;</code> | base byte |
| 73 | <code>&#x27;c&#x27;</code> | base byte |
| 74 | <code>&#x27;d&#x27;</code> | base byte |
| 75 | <code>&#x27;e&#x27;</code> | base byte |
| 76 | <code>&#x27;f&#x27;</code> | base byte |
| 77 | <code>&#x27;g&#x27;</code> | base byte |
| 78 | <code>&#x27;h&#x27;</code> | base byte |
| 79 | <code>&#x27;i&#x27;</code> | base byte |
| 80 | <code>&#x27;j&#x27;</code> | base byte |
| 81 | <code>&#x27;k&#x27;</code> | base byte |
| 82 | <code>&#x27;l&#x27;</code> | base byte |
| 83 | <code>&#x27;m&#x27;</code> | base byte |
| 84 | <code>&#x27;n&#x27;</code> | base byte |
| 85 | <code>&#x27;o&#x27;</code> | base byte |
| 86 | <code>&#x27;p&#x27;</code> | base byte |
| 87 | <code>&#x27;q&#x27;</code> | base byte |
| 88 | <code>&#x27;r&#x27;</code> | base byte |
| 89 | <code>&#x27;s&#x27;</code> | base byte |
| 90 | <code>&#x27;t&#x27;</code> | base byte |
| 91 | <code>&#x27;u&#x27;</code> | base byte |
| 92 | <code>&#x27;v&#x27;</code> | base byte |
| 93 | <code>&#x27;w&#x27;</code> | base byte |
| 94 | <code>&#x27;x&#x27;</code> | base byte |
| 95 | <code>&#x27;y&#x27;</code> | base byte |
| 96 | <code>&#x27;z&#x27;</code> | base byte |
| 97 | <code>&#x27;{&#x27;</code> | base byte |
| 98 | <code>&#x27;&#124;&#x27;</code> | base byte |
| 99 | <code>&#x27;}&#x27;</code> | base byte |
| 100 | <code>&#x27;~&#x27;</code> | base byte |
| 101 | <code>&#x27;u_&#x27;</code> | learned merge |
| 102 | <code>&#x27;{x&#x27;</code> | learned merge |
| 103 | <code>&#x27;=0&#x27;</code> | learned merge |
| 104 | <code>&#x27;{x}&#x27;</code> | learned merge |
| 105 | <code>&#x27;\\c&#x27;</code> | learned merge |
| 106 | <code>&#x27;do&#x27;</code> | learned merge |
| 107 | <code>&#x27;\\cdo&#x27;</code> | learned merge |
| 108 | <code>&#x27;\\cdot&#x27;</code> | learned merge |
| 109 | <code>&#x27;\\cdot &#x27;</code> | learned merge |
| 110 | <code>&#x27;u_{x}&#x27;</code> | learned merge |
| 111 | <code>&#x27;u_{x})&#x27;</code> | learned merge |
| 112 | <code>&#x27;xx&#x27;</code> | learned merge |
| 113 | <code>&#x27;(u_{x})&#x27;</code> | learned merge |
| 114 | <code>&#x27;(u&#x27;</code> | learned merge |
| 115 | <code>&#x27;t}&#x27;</code> | learned merge |
| 116 | <code>&#x27;2}&#x27;</code> | learned merge |
| 117 | <code>&#x27;{2}&#x27;</code> | learned merge |
| 118 | <code>&#x27;^{2}&#x27;</code> | learned merge |
| 119 | <code>&#x27;(u)&#x27;</code> | learned merge |
| 120 | <code>&#x27;xx}&#x27;</code> | learned merge |
| 121 | <code>&#x27;{t}&#x27;</code> | learned merge |
| 122 | <code>&#x27;{xx}&#x27;</code> | learned merge |
| 123 | <code>&#x27;u_{xx}&#x27;</code> | learned merge |
| 124 | <code>&#x27;(u_{xx}&#x27;</code> | learned merge |
| 125 | <code>&#x27;(u_{xx})&#x27;</code> | learned merge |
| 126 | <code>&#x27;u_{t}&#x27;</code> | learned merge |
| 127 | <code>&#x27;(u_{t}&#x27;</code> | learned merge |
| 128 | <code>&#x27;(u_{t})&#x27;</code> | learned merge |
| 129 | <code>&#x27;y}&#x27;</code> | learned merge |
| 130 | <code>&#x27;{y}&#x27;</code> | learned merge |
| 131 | <code>&#x27;)+&#x27;</code> | learned merge |
| 132 | <code>&#x27;u}&#x27;</code> | learned merge |
| 133 | <code>&#x27;{u}&#x27;</code> | learned merge |
| 134 | <code>&#x27;)+(&#x27;</code> | learned merge |
| 135 | <code>&#x27;u_{y}&#x27;</code> | learned merge |
| 136 | <code>&#x27;(x&#x27;</code> | learned merge |
| 137 | <code>&#x27;(x)&#x27;</code> | learned merge |
| 138 | <code>&#x27;+(u_{t})&#x27;</code> | learned merge |
| 139 | <code>&#x27;(u_{y}&#x27;</code> | learned merge |
| 140 | <code>&#x27;\\cdot (u_{x})&#x27;</code> | learned merge |
| 141 | <code>&#x27;yy&#x27;</code> | learned merge |
| 142 | <code>&#x27;(u_{y})&#x27;</code> | learned merge |
| 143 | <code>&#x27;+(u_{xx})&#x27;</code> | learned merge |
| 144 | <code>&#x27;(u_{x})\\cdot &#x27;</code> | learned merge |
| 145 | <code>&#x27;{yy&#x27;</code> | learned merge |
| 146 | <code>&#x27;{yy}&#x27;</code> | learned merge |
| 147 | <code>&#x27;u_{yy}&#x27;</code> | learned merge |
| 148 | <code>&#x27;(u_{yy}&#x27;</code> | learned merge |
| 149 | <code>&#x27;(u_{yy})&#x27;</code> | learned merge |
| 150 | <code>&#x27;xxx&#x27;</code> | learned merge |
| 151 | <code>&#x27;tt&#x27;</code> | learned merge |
| 152 | <code>&#x27;{tt&#x27;</code> | learned merge |
| 153 | <code>&#x27;{u}^{2}&#x27;</code> | learned merge |
| 154 | <code>&#x27;{tt}&#x27;</code> | learned merge |
| 155 | <code>&#x27;u_{tt}&#x27;</code> | learned merge |
| 156 | <code>&#x27;(u_{tt}&#x27;</code> | learned merge |
| 157 | <code>&#x27;(u_{tt})&#x27;</code> | learned merge |
| 158 | <code>&#x27;({u}^{2}&#x27;</code> | learned merge |
| 159 | <code>&#x27;)=0&#x27;</code> | learned merge |
| 160 | <code>&#x27;({u}^{2})&#x27;</code> | learned merge |
| 161 | <code>&#x27;+(u)&#x27;</code> | learned merge |
| 162 | <code>&#x27;y)&#x27;</code> | learned merge |
| 163 | <code>&#x27;(y)&#x27;</code> | learned merge |
| 164 | <code>&#x27;{x}^{2}&#x27;</code> | learned merge |
| 165 | <code>&#x27;xxx}&#x27;</code> | learned merge |
| 166 | <code>&#x27;({x}^{2}&#x27;</code> | learned merge |
| 167 | <code>&#x27;{u_{x}&#x27;</code> | learned merge |
| 168 | <code>&#x27;{xx&#x27;</code> | learned merge |
| 169 | <code>&#x27;{u_{x}}&#x27;</code> | learned merge |
| 170 | <code>&#x27;+(u_{yy})&#x27;</code> | learned merge |
| 171 | <code>&#x27;({x}^{2})&#x27;</code> | learned merge |
| 172 | <code>&#x27;(u_{t})+&#x27;</code> | learned merge |
| 173 | <code>&#x27;{u_{x}}^{2}&#x27;</code> | learned merge |
| 174 | <code>&#x27;({u_{x}}^{2}&#x27;</code> | learned merge |
| 175 | <code>&#x27;({u_{x}}^{2})&#x27;</code> | learned merge |
| 176 | <code>&#x27;rt&#x27;</code> | learned merge |
| 177 | <code>&#x27;tt}&#x27;</code> | learned merge |
| 178 | <code>&#x27;+(u_{tt})&#x27;</code> | learned merge |
| 179 | <code>&#x27;+(u_{x})&#x27;</code> | learned merge |
| 180 | <code>&#x27;\\p&#x27;</code> | learned merge |
| 181 | <code>&#x27;al&#x27;</code> | learned merge |
| 182 | <code>&#x27;art&#x27;</code> | learned merge |
| 183 | <code>&#x27;ial&#x27;</code> | learned merge |
| 184 | <code>&#x27;{xxx}&#x27;</code> | learned merge |
| 185 | <code>&#x27;u_{xxx}&#x27;</code> | learned merge |
| 186 | <code>&#x27;(u_{xxx}&#x27;</code> | learned merge |
| 187 | <code>&#x27;\\part&#x27;</code> | learned merge |
| 188 | <code>&#x27;(u_{xxx})&#x27;</code> | learned merge |
| 189 | <code>&#x27;\\partial&#x27;</code> | learned merge |
| 190 | <code>&#x27;\\partial_&#x27;</code> | learned merge |
| 191 | <code>&#x27;(u_{x})\\cdot (u)&#x27;</code> | learned merge |
| 192 | <code>&#x27;(u)\\cdot (u_{x})&#x27;</code> | learned merge |
| 193 | <code>&#x27;\\cdot (u_{y})&#x27;</code> | learned merge |
| 194 | <code>&#x27;(u_{y})\\cdot &#x27;</code> | learned merge |
| 195 | <code>&#x27;{y}^{2}&#x27;</code> | learned merge |
| 196 | <code>&#x27;({y}^{2}&#x27;</code> | learned merge |
| 197 | <code>&#x27;({y}^{2})&#x27;</code> | learned merge |
| 198 | <code>&#x27;{(u&#x27;</code> | learned merge |
| 199 | <code>&#x27;\\cdot u_{x})&#x27;</code> | learned merge |
| 200 | <code>&#x27;\\cdot u_{x})}&#x27;</code> | learned merge |
| 201 | <code>&#x27;De&#x27;</code> | learned merge |
| 202 | <code>&#x27;\\De&#x27;</code> | learned merge |
| 203 | <code>&#x27;lt&#x27;</code> | learned merge |
| 204 | <code>&#x27;{(u\\cdot u_{x})}&#x27;</code> | learned merge |
| 205 | <code>&#x27;\\Delt&#x27;</code> | learned merge |
| 206 | <code>&#x27;\\Delta&#x27;</code> | learned merge |
| 207 | <code>&#x27;xy&#x27;</code> | learned merge |
| 208 | <code>&#x27;{xy&#x27;</code> | learned merge |
| 209 | <code>&#x27;{u}^&#x27;</code> | learned merge |
| 210 | <code>&#x27;{xy}&#x27;</code> | learned merge |
| 211 | <code>&#x27;u_{xy}&#x27;</code> | learned merge |
| 212 | <code>&#x27;(u_{xy}&#x27;</code> | learned merge |
| 213 | <code>&#x27;(u_{xy})&#x27;</code> | learned merge |
| 214 | <code>&#x27;\\cdot (x)&#x27;</code> | learned merge |
| 215 | <code>&#x27;u_{tt})+(&#x27;</code> | learned merge |
| 216 | <code>&#x27;+({u}^{2})&#x27;</code> | learned merge |
| 217 | <code>&#x27;\\cdot (u)&#x27;</code> | learned merge |
| 218 | <code>&#x27;3}&#x27;</code> | learned merge |
| 219 | <code>&#x27;{3}&#x27;</code> | learned merge |
| 220 | <code>&#x27;xxt&#x27;</code> | learned merge |
| 221 | <code>&#x27;xxxx&#x27;</code> | learned merge |
| 222 | <code>&#x27;\\partial_{x}&#x27;</code> | learned merge |
| 223 | <code>&#x27;+({u_{x}}^{2})&#x27;</code> | learned merge |
| 224 | <code>&#x27;{xxt&#x27;</code> | learned merge |
| 225 | <code>&#x27;\\Delta{u}&#x27;</code> | learned merge |
| 226 | <code>&#x27;{u}^{3}&#x27;</code> | learned merge |
| 227 | <code>&#x27;\\partial_{x}{(u\\cdot u_{x})}&#x27;</code> | learned merge |
| 228 | <code>&#x27;{xxt}&#x27;</code> | learned merge |
| 229 | <code>&#x27;u_{xxt}&#x27;</code> | learned merge |
| 230 | <code>&#x27;(u_{xxt}&#x27;</code> | learned merge |
| 231 | <code>&#x27;(u_{xxt})&#x27;</code> | learned merge |
| 232 | <code>&#x27;+(u)\\cdot (u_{x})&#x27;</code> | learned merge |
| 233 | <code>&#x27;(u_{x})/&#x27;</code> | learned merge |
| 234 | <code>&#x27;\\cdot (u_{xx})&#x27;</code> | learned merge |
| 235 | <code>&#x27;\\s&#x27;</code> | learned merge |
| 236 | <code>&#x27;(u_{x})/(x)&#x27;</code> | learned merge |
| 237 | <code>&#x27;+(u_{x})\\cdot (u)&#x27;</code> | learned merge |
| 238 | <code>&#x27;+(u_{xxx})&#x27;</code> | learned merge |
| 239 | <code>&#x27;u_{xx})+(&#x27;</code> | learned merge |
| 240 | <code>&#x27;({u}^{3}&#x27;</code> | learned merge |
| 241 | <code>&#x27;{xxxx&#x27;</code> | learned merge |
| 242 | <code>&#x27;{xxxx}&#x27;</code> | learned merge |
| 243 | <code>&#x27;u_{xxxx}&#x27;</code> | learned merge |
| 244 | <code>&#x27;(u_{xxxx}&#x27;</code> | learned merge |
| 245 | <code>&#x27;(u_{xxxx})&#x27;</code> | learned merge |
| 246 | <code>&#x27;in&#x27;</code> | learned merge |
| 247 | <code>&#x27;xt&#x27;</code> | learned merge |
| 248 | <code>&#x27;zz&#x27;</code> | learned merge |
| 249 | <code>&#x27;{u_{y}&#x27;</code> | learned merge |
| 250 | <code>&#x27;{xt&#x27;</code> | learned merge |
| 251 | <code>&#x27;{zz&#x27;</code> | learned merge |
| 252 | <code>&#x27;\\partial_{xx}&#x27;</code> | learned merge |
| 253 | <code>&#x27;\\sin&#x27;</code> | learned merge |
| 254 | <code>&#x27;({u}^{3})&#x27;</code> | learned merge |
| 255 | <code>&#x27;{u_{y}}&#x27;</code> | learned merge |
| 256 | <code>&#x27;{xt}&#x27;</code> | learned merge |
| 257 | <code>&#x27;u_{xt}&#x27;</code> | learned merge |
| 258 | <code>&#x27;(u_{xt}&#x27;</code> | learned merge |
| 259 | <code>&#x27;{zz}&#x27;</code> | learned merge |
| 260 | <code>&#x27;u_{zz}&#x27;</code> | learned merge |
| 261 | <code>&#x27;(u_{zz}&#x27;</code> | learned merge |
| 262 | <code>&#x27;{u_{y}}^{2}&#x27;</code> | learned merge |
| 263 | <code>&#x27;({u_{y}}^{2}&#x27;</code> | learned merge |
| 264 | <code>&#x27;(u_{xt})&#x27;</code> | learned merge |
| 265 | <code>&#x27;(u_{zz})&#x27;</code> | learned merge |
| 266 | <code>&#x27;({u_{y}}^{2})&#x27;</code> | learned merge |
| 267 | <code>&#x27;(\\Delta{u}&#x27;</code> | learned merge |
| 268 | <code>&#x27;+(u_{xxt})&#x27;</code> | learned merge |
| 269 | <code>&#x27;(u_{tt})+(&#x27;</code> | learned merge |
| 270 | <code>&#x27;+(u_{y})&#x27;</code> | learned merge |
| 271 | <code>&#x27;+(u_{x})/(x)&#x27;</code> | learned merge |
| 272 | <code>&#x27;\\partial_{x}{(u\\cdot u_{x})})+(&#x27;</code> | learned merge |
| 273 | <code>&#x27;(t&#x27;</code> | learned merge |
| 274 | <code>&#x27;(t)&#x27;</code> | learned merge |
| 275 | <code>&#x27;\\cdot (y)&#x27;</code> | learned merge |
| 276 | <code>&#x27;(u)\\cdot &#x27;</code> | learned merge |
| 277 | <code>&#x27;(x)\\cdot (u_{x})&#x27;</code> | learned merge |
| 278 | <code>&#x27;+(u_{zz})&#x27;</code> | learned merge |
| 279 | <code>&#x27;({u}^{2})\\cdot (u_{x})&#x27;</code> | learned merge |
| 280 | <code>&#x27;yy}&#x27;</code> | learned merge |
| 281 | <code>&#x27;(u_{x})\\cdot ({u}^{2})&#x27;</code> | learned merge |
| 282 | <code>&#x27;(u_{x})\\cdot (x)&#x27;</code> | learned merge |
| 283 | <code>&#x27;u_{tt})=0&#x27;</code> | learned merge |
| 284 | <code>&#x27;+(u_{xxxx})&#x27;</code> | learned merge |
| 285 | <code>&#x27;(u_{y})\\cdot (y)&#x27;</code> | learned merge |
| 286 | <code>&#x27;(u)+&#x27;</code> | learned merge |
| 287 | <code>&#x27;(u_{xx})+&#x27;</code> | learned merge |
| 288 | <code>&#x27;(y)\\cdot (u_{y})&#x27;</code> | learned merge |
| 289 | <code>&#x27;{u_{t}&#x27;</code> | learned merge |
| 290 | <code>&#x27;{u_{t}}&#x27;</code> | learned merge |
| 291 | <code>&#x27;\\cdot ({x}^{2})&#x27;</code> | learned merge |
| 292 | <code>&#x27;(u_{x})+&#x27;</code> | learned merge |
| 293 | <code>&#x27;+(u_{xt})&#x27;</code> | learned merge |
| 294 | <code>&#x27;(u_{xx})\\cdot (u_{x})&#x27;</code> | learned merge |
| 295 | <code>&#x27;(u_{xx})+(u_{yy})&#x27;</code> | learned merge |
| 296 | <code>&#x27;(u_{xx})+(&#x27;</code> | learned merge |
| 297 | <code>&#x27;/({x}^{2})&#x27;</code> | learned merge |
| 298 | <code>&#x27;(\\partial_{xx}&#x27;</code> | learned merge |
| 299 | <code>&#x27;+({u}^{3})&#x27;</code> | learned merge |
| 300 | <code>&#x27;(u_{tt})+(u_{xx})&#x27;</code> | learned merge |
| 301 | <code>&#x27;\\cdot (u_{yy})&#x27;</code> | learned merge |
| 302 | <code>&#x27;u_{xxxx})+(&#x27;</code> | learned merge |
| 303 | <code>&#x27;(u_{xx})+(u_{tt})&#x27;</code> | learned merge |
| 304 | <code>&#x27;(u_{x})\\cdot (u_{xx})&#x27;</code> | learned merge |
| 305 | <code>&#x27;+({u}^{2})\\cdot (u_{x})&#x27;</code> | learned merge |
| 306 | <code>&#x27;(u_{x})\\cdot (y)&#x27;</code> | learned merge |
| 307 | <code>&#x27;u_{xx})=0&#x27;</code> | learned merge |
| 308 | <code>&#x27;\\cdot ({y}^{2})&#x27;</code> | learned merge |
| 309 | <code>&#x27;)+(u_{tt})=0&#x27;</code> | learned merge |
| 310 | <code>&#x27;(u_{yy})+(u_{xx})&#x27;</code> | learned merge |
| 311 | <code>&#x27;u_{xt})+(&#x27;</code> | learned merge |
| 312 | <code>&#x27;\\e&#x27;</code> | learned merge |
| 313 | <code>&#x27;xp&#x27;</code> | learned merge |
| 314 | <code>&#x27;(u)/&#x27;</code> | learned merge |
| 315 | <code>&#x27;(u_{yy})/({x}^{2})&#x27;</code> | learned merge |
| 316 | <code>&#x27;{xxtt}&#x27;</code> | learned merge |
| 317 | <code>&#x27;u_{xxtt}&#x27;</code> | learned merge |
| 318 | <code>&#x27;xxxxx&#x27;</code> | learned merge |
| 319 | <code>&#x27;{xxxxx&#x27;</code> | learned merge |
| 320 | <code>&#x27;\\sin{u}&#x27;</code> | learned merge |
| 321 | <code>&#x27;{u_{t}}^{2}&#x27;</code> | learned merge |
| 322 | <code>&#x27;({u_{t}}^{2}&#x27;</code> | learned merge |
| 323 | <code>&#x27;\\exp&#x27;</code> | learned merge |
| 324 | <code>&#x27;{xxxxx}&#x27;</code> | learned merge |
| 325 | <code>&#x27;u_{xxxxx}&#x27;</code> | learned merge |
| 326 | <code>&#x27;(u_{xxxxx}&#x27;</code> | learned merge |
| 327 | <code>&#x27;({u_{t}}^{2})&#x27;</code> | learned merge |
| 328 | <code>&#x27;(u_{xxxxx})&#x27;</code> | learned merge |
| 329 | <code>&#x27;)+(u_{t})&#x27;</code> | learned merge |
| 330 | <code>&#x27;+(x)\\cdot (u_{x})&#x27;</code> | learned merge |
| 331 | <code>&#x27;\\cdot ({u_{x}}^{2})&#x27;</code> | learned merge |
| 332 | <code>&#x27;(x)\\cdot (y)&#x27;</code> | learned merge |
| 333 | <code>&#x27;+({x}^{2})&#x27;</code> | learned merge |
| 334 | <code>&#x27;(u_{y})\\cdot (u)&#x27;</code> | learned merge |
| 335 | <code>&#x27;(u)\\cdot (u_{y})&#x27;</code> | learned merge |
| 336 | <code>&#x27;(u_{xxx})\\cdot (u)&#x27;</code> | learned merge |
| 337 | <code>&#x27;+(y)\\cdot (u_{y})&#x27;</code> | learned merge |
| 338 | <code>&#x27;+(u_{x})\\cdot (u_{xx})&#x27;</code> | learned merge |
| 339 | <code>&#x27;\\cdot (u_{xy})&#x27;</code> | learned merge |
| 340 | <code>&#x27;u_{yy})+(&#x27;</code> | learned merge |
| 341 | <code>&#x27;\\partial_{x}{(u\\cdot u_{x})})=0&#x27;</code> | learned merge |
| 342 | <code>&#x27;(u_{t})+(u_{xx})&#x27;</code> | learned merge |
| 343 | <code>&#x27;(y)\\cdot (u_{x})&#x27;</code> | learned merge |
| 344 | <code>&#x27;)\\cdot &#x27;</code> | learned merge |
| 345 | <code>&#x27;(u_{t})+(u_{x})&#x27;</code> | learned merge |
| 346 | <code>&#x27;({u_{x}}^{2})+&#x27;</code> | learned merge |
| 347 | <code>&#x27;(u_{y})\\cdot (x)&#x27;</code> | learned merge |
| 348 | <code>&#x27;+(u_{xy})&#x27;</code> | learned merge |
| 349 | <code>&#x27;(u_{xx})\\cdot (u)&#x27;</code> | learned merge |
