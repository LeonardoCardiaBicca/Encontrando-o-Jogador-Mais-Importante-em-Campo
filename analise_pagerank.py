###Inicio###
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
import sys

try:
    from statsbombpy import sb
except ImportError:
    print("\n[AVISO] A biblioteca 'statsbombpy' não foi encontrada neste ambiente.")
    print("Para instalá-la, execute um dos comandos abaixo no terminal:\n")
    print("    pip install statsbombpy")
    print("    # ou, se 'pip' não for reconhecido:")
    print("    python -m pip install statsbombpy\n")
    print("Após a instalação, execute o programa novamente.")
    sys.exit(1)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator

try:
    from mplsoccer import Pitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15
})

COR_FUNDO_CAMPO = "#F8FAFC"
COR_LINHAS_CAMPO = "#475569"
COR_AZUL_PROFUNDO = "#003893"
COR_CELESTE_ARG = "#43A1D5"
COR_AZUL_CLARO_ARG = "#93C5FD"
COR_DOURADO_SOL = "#D97706"
COR_CINZA_TUFTE = "#64748B"
COR_TEXTO_ESCURO = "#0F172A"

MAPA_NOMES_CURTOS = {
    "Damián Emiliano Martínez": "E. Martínez",
    "Nahuel Molina Lucero": "Molina",
    "Gonzalo Ariel Montiel": "Montiel",
    "Cristian Gabriel Romero": "C. Romero",
    "Nicolás Hernán Otamendi": "Otamendi",
    "Lisandro Martínez": "L. Martínez",
    "Germán Alejandro Pezzella": "Pezzella",
    "Juan Marcos Foyth": "Foyth",
    "Nicolás Alejandro Tagliafico": "Tagliafico",
    "Marcos Javier Acuña": "Acuña",
    "Enzo Fernandez": "Enzo Fernández",
    "Rodrigo Javier De Paul": "De Paul",
    "Alexis Mac Allister": "Mac Allister",
    "Leandro Daniel Paredes": "Paredes",
    "Guido Rodríguez": "G. Rodríguez",
    "Exequiel Alejandro Palacios": "Palacios",
    "Alejandro Darío Gómez": "Papu Gómez",
    "Thiago Ezequiel Almada": "Almada",
    "Lionel Andrés Messi Cuccittini": "Messi",
    "Julián Álvarez": "J. Álvarez",
    "Ángel Fabián Di María Hernández": "Di María",
    "Lautaro Javier Martínez": "Lautaro Martínez",
    "Paulo Bruno Exequiel Dybala": "Dybala",
    "Ángel Martín Correa": "A. Correa"
}


def abreviar_nome(nome_completo):
    """Retorna o nome tático curto do jogador para rótulos visuais limpos."""
    return MAPA_NOMES_CURTOS.get(nome_completo, nome_completo.split()[-1])


TIME_ALVO = "Argentina"

PASTA_FIGURAS = "figuras"  # todas as figuras geradas são salvas aqui


def _garantir_pasta(*caminhos):
    """Cria a pasta de destino de cada caminho informado, se ainda não existir."""
    for caminho in caminhos:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)

IDS_JOGOS = [
    3857300,  # Arábia Saudita (Fase de Grupos)
    3857289,  # México (Fase de Grupos)
    3857264,  # Polônia (Fase de Grupos)
    3869151,  # Austrália (Oitavas de Final)
    3869321,  # Países Baixos (Quartas de Final)
    3869519,  # Croácia (Semifinal)
    3869685,  # França (Final)
]

NOMES_JOGOS = {
    3857300: "Arábia Saudita",
    3857289: "México",
    3857264: "Polônia",
    3869151: "Austrália",
    3869321: "Países Baixos",
    3869519: "Croácia",
    3869685: "França",
}

INVERTER_GRAFO = True

RECORTE_ATE_PRIMEIRA_SUBSTITUICAO = False

MODO_ALPHA = "fixo"
ALPHA_FIXO = 0.85

# Matriz oficial 8x12 de Expected Threat (Karun Singh)
XT_GRID = np.array([
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.01745060, 0.02122129, 0.02756312, 0.03485072, 0.03792590],
    [0.00750072, 0.00878589, 0.00942382, 0.01059490, 0.01214719, 0.01384540, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00887990, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.02412240, 0.02855202, 0.05491138, 0.06442595],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.01997070, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.01997070, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.00887990, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.02412240, 0.02855202, 0.05491138, 0.06442595],
    [0.00750072, 0.00878589, 0.00942382, 0.01059490, 0.01214719, 0.01384540, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.01745060, 0.02122129, 0.02756312, 0.03485072, 0.03792590]
])

CONSTANTE_BASE_XT = float(np.median(XT_GRID))

CONSTANTE_BASE_XG = 0.05
FATOR_POSSE_XG = 0.50


def obter_xt(x, y):
    """Converte coordenadas StatsBomb (0..120, 0..80) nos índices da grade 8x12."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    col = np.clip((x_arr / 120.0 * 12).astype(int), 0, 11)
    lin = np.clip((y_arr / 80.0 * 8).astype(int), 0, 7)
    return XT_GRID[lin, col]


@lru_cache(maxsize=64)
def carregar_eventos(match_id):
    """Baixa e armazena em cache os eventos da partida via StatsBomb Open Data."""
    return sb.events(match_id=int(match_id))


def calcular_xg_futuro_da_posse(passes, chutes):
    """
    Soma exclusivamente o xG dos chutes posteriores ao passe, ocorridos na mesma
    posse de bola (possession) e no mesmo período (period).
    """
    creditos = pd.Series(0.0, index=passes.index, dtype=float)
    for (periodo, posse), bloco in passes.groupby(["period", "possession"]):
        tiros = chutes.loc[
            (chutes["period"] == periodo) & (chutes["possession"] == posse)
        ].sort_values("index")
        if tiros.empty:
            continue
        indices_chutes = tiros["index"].to_numpy()
        valores_xg = tiros["shot_statsbomb_xg"].to_numpy(dtype=float)
        acumulado_futuro = np.r_[np.cumsum(valores_xg[::-1])[::-1], 0.0]
        posicoes = np.searchsorted(indices_chutes, bloco["index"].to_numpy(), side="right")
        creditos.loc[bloco.index] = acumulado_futuro[posicoes]
    return creditos


def obter_eventos_titulares(match_id, time_alvo, recorte_ate_sub=None):
    """
    Filtra os eventos da equipe até o final da prorrogação (period <= 4,
    excluindo disputa por pênaltis) e aplica opcionalmente o recorte até a
    primeira substituição.
    """
    if recorte_ate_sub is None:
        recorte_ate_sub = RECORTE_ATE_PRIMEIRA_SUBSTITUICAO

    eventos = carregar_eventos(match_id)
    eventos_time = eventos[
        (eventos["team"] == time_alvo) & (eventos["period"] <= 4)
    ].sort_values(by=["period", "minute", "second", "index"]).copy()

    subs = eventos_time[eventos_time["type"] == "Substitution"]
    if recorte_ate_sub and not subs.empty:
        primeira_sub_idx = subs.iloc[0]["index"]
        eventos_recorte = eventos_time[eventos_time["index"] < primeira_sub_idx].copy()
    else:
        eventos_recorte = eventos_time.copy()

    return eventos_recorte


def estimar_alpha_empirico(eventos_recorte):
    """
    Estima empiricamente a probabilidade de retenção/continuidade da cadeia de
    passes, garantindo 0.01 <= alpha <= 0.99.
    """
    passes_completos = eventos_recorte[
        (eventos_recorte["type"] == "Pass") &
        (eventos_recorte["pass_outcome"].isna())
    ]
    n_continuacoes = len(passes_completos)

    passes_errados = eventos_recorte[
        (eventos_recorte["type"] == "Pass") &
        (eventos_recorte["pass_outcome"].notna())
    ]
    chutes = eventos_recorte[eventos_recorte["type"] == "Shot"]
    perdas = eventos_recorte[eventos_recorte["type"].isin(["Dispossessed", "Miscontrol"])]

    n_encerramentos = len(passes_errados) + len(chutes) + len(perdas)
    total = n_continuacoes + n_encerramentos
    if total == 0:
        return ALPHA_FIXO

    alpha = n_continuacoes / total
    return float(np.clip(alpha, 0.01, 0.99))


def obter_df_passes(match_id, time_alvo, recorte_ate_sub=None, verbose=False):
    """
    Extrai os passes completos válidos da partida e constrói todas as métricas
    de ponderação tática:
    - peso_tatico (avanço ao gol, assistência para chute e retenção sob pressão)
    - peso_xt (piso mediano da grade xT + ganho posicional positivo Delta xT)
    - peso_xg (piso mediano de chutes + xA direto validado + fator * xG futuro da posse)
    """
    if recorte_ate_sub is None:
        recorte_ate_sub = RECORTE_ATE_PRIMEIRA_SUBSTITUICAO

    eventos_recorte = obter_eventos_titulares(
        match_id, time_alvo, recorte_ate_sub=recorte_ate_sub
    )

    for coluna in ["pass_outcome", "under_pressure", "pass_shot_assist", "pass_assisted_shot_id"]:
        if coluna not in eventos_recorte.columns:
            eventos_recorte[coluna] = np.nan

    passes = eventos_recorte[
        (eventos_recorte["type"] == "Pass") &
        (eventos_recorte["pass_outcome"].isna())
    ].dropna(subset=["player", "pass_recipient", "location", "pass_end_location"]).copy()

    if passes.empty:
        raise ValueError(f"Sem passes válidos para {time_alvo} na partida {match_id}.")

    passes[["x_inicio", "y_inicio"]] = pd.DataFrame(
        passes["location"].map(lambda p: p[:2]).tolist(), index=passes.index
    )
    passes[["x_fim", "y_fim"]] = pd.DataFrame(
        passes["pass_end_location"].map(lambda p: p[:2]).tolist(), index=passes.index
    )
    passes["delta_x"] = passes["x_fim"] - passes["x_inicio"]
    passes["avanco_positivo"] = passes["delta_x"].clip(lower=0.0)

    dist_inicio = np.hypot(120.0 - passes["x_inicio"], 40.0 - passes["y_inicio"])
    dist_fim = np.hypot(120.0 - passes["x_fim"], 40.0 - passes["y_fim"])
    passes["ganho_proximidade_gol"] = (dist_inicio - dist_fim).clip(lower=0.0)

    passes["sob_pressao"] = passes["under_pressure"].fillna(False).astype(int)
    passes["passe_para_chute"] = passes["pass_shot_assist"].fillna(False).astype(int)
    passes["peso_tatico"] = (
        1.0
        + (passes["ganho_proximidade_gol"] / 10.0)
        + (passes["passe_para_chute"] * 5.0)
    ) * np.where(passes["sob_pressao"] == 1, 1.5, 1.0)

    xt_inicio = obter_xt(passes["x_inicio"], passes["y_inicio"])
    xt_fim = obter_xt(passes["x_fim"], passes["y_fim"])
    passes["delta_xt"] = xt_fim - xt_inicio
    passes["peso_xt"] = CONSTANTE_BASE_XT + passes["delta_xt"].clip(lower=0.0)

    chutes = eventos_recorte[eventos_recorte["type"] == "Shot"][
        ["id", "period", "index", "possession", "shot_statsbomb_xg"]
    ].copy()
    chutes["shot_statsbomb_xg"] = (
        pd.to_numeric(chutes["shot_statsbomb_xg"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    mapa_xa = chutes.set_index("id")["shot_statsbomb_xg"].to_dict()
    mapa_periodo = chutes.set_index("id")["period"].to_dict()
    mapa_posse = chutes.set_index("id")["possession"].to_dict()
    mapa_indice = chutes.set_index("id")["index"].to_dict()

    if "pass_assisted_shot_id" in passes.columns:
        passes["xa_direto"] = passes["pass_assisted_shot_id"].map(mapa_xa).fillna(0.0)
        valido = (
            passes["period"].eq(passes["pass_assisted_shot_id"].map(mapa_periodo))
            & passes["possession"].eq(passes["pass_assisted_shot_id"].map(mapa_posse))
            & (passes["index"] < passes["pass_assisted_shot_id"].map(mapa_indice))
        )
        passes.loc[~valido, "xa_direto"] = 0.0
    else:
        passes["xa_direto"] = 0.0

    passes["xg_posse"] = calcular_xg_futuro_da_posse(passes, chutes)
    passes["peso_xg"] = (
        CONSTANTE_BASE_XG
        + passes["xa_direto"]
        + (FATOR_POSSE_XG * passes["xg_posse"])
    )

    colunas_passes = [
        "period", "minute", "second", "index", "possession",
        "player", "pass_recipient",
        "x_inicio", "y_inicio", "x_fim", "y_fim",
        "delta_x", "avanco_positivo", "ganho_proximidade_gol",
        "sob_pressao", "passe_para_chute", "peso_tatico",
        "delta_xt", "xa_direto", "xg_posse", "peso_xt", "peso_xg"
    ]
    df_passes = passes[colunas_passes].reset_index(drop=True)

    if verbose:
        n_jogadores = len(set(df_passes["player"]) | set(df_passes["pass_recipient"]))
        print(f"Partida {match_id}: {len(df_passes)} passes válidos | {n_jogadores} jogadores no grafo.")

    return df_passes


def calibrar_parametros_xg(lista_match_ids, time_alvo, recorte_ate_sub=None):
    """
    Calibra empiricamente CONSTANTE_BASE_XG (mediana do xG dos chutes) e
    FATOR_POSSE_XG (razão entre soma de xA direto e soma de xG futuro das posses)
    ao longo de todas as partidas analisadas.
    """
    global CONSTANTE_BASE_XG, FATOR_POSSE_XG

    bases = [
        obter_df_passes(match_id, time_alvo, recorte_ate_sub=recorte_ate_sub, verbose=False)
        for match_id in lista_match_ids
    ]
    xg_chutes = []
    for match_id in lista_match_ids:
        eventos = obter_eventos_titulares(match_id, time_alvo, recorte_ate_sub=recorte_ate_sub)
        tiros = eventos.loc[eventos["type"] == "Shot", "shot_statsbomb_xg"]
        xg_chutes.extend(
            pd.to_numeric(tiros, errors="coerce").dropna().loc[lambda s: s > 0].tolist()
        )

    base_xg = float(np.median(xg_chutes)) if xg_chutes else 1.0
    total_direto = sum(base["xa_direto"].sum() for base in bases)
    total_posse = sum(base["xg_posse"].sum() for base in bases)
    fator_posse = float(total_direto / total_posse) if total_posse > 0 else 0.0

    CONSTANTE_BASE_XG = base_xg
    FATOR_POSSE_XG = fator_posse
    return base_xg, fator_posse


def calcular_pagerank_futebol(
    MATCH_ID,
    TIME_ALVO,
    coluna_peso=None,
    vetor_personalizacao=None,
    inverter_grafo=None,
    recorte_ate_sub=None,
    modo_alpha=None
):
    """
    Constrói o dígrafo ponderado da partida (invertendo as arestas quando
    inverter_grafo=True para medir o PageRank do fornecedor/distribuidor) e
    calcula o vetor estacionário de PageRank.
    """
    if inverter_grafo is None:
        inverter_grafo = INVERTER_GRAFO
    if recorte_ate_sub is None:
        recorte_ate_sub = RECORTE_ATE_PRIMEIRA_SUBSTITUICAO
    if modo_alpha is None:
        modo_alpha = MODO_ALPHA

    df_passes = obter_df_passes(
        MATCH_ID, TIME_ALVO, recorte_ate_sub=recorte_ate_sub, verbose=False
    )
    alpha = (
        estimar_alpha_empirico(
            obter_eventos_titulares(MATCH_ID, TIME_ALVO, recorte_ate_sub=recorte_ate_sub)
        )
        if modo_alpha == "empirico" else ALPHA_FIXO
    )

    if coluna_peso is None:
        arestas = (
            df_passes.groupby(["player", "pass_recipient"])
            .size()
            .reset_index(name="peso_final")
        )
    else:
        if (df_passes[coluna_peso] < 0).any():
            raise ValueError(
                f"A coluna '{coluna_peso}' contém valores negativos, violando a Cadeia de Markov."
            )
        arestas = (
            df_passes.groupby(["player", "pass_recipient"])[coluna_peso]
            .sum()
            .reset_index(name="peso_final")
        )

    G = nx.DiGraph()
    for _, row in arestas.iterrows():
        if inverter_grafo:
            G.add_edge(row["pass_recipient"], row["player"], weight=row["peso_final"])
        else:
            G.add_edge(row["player"], row["pass_recipient"], weight=row["peso_final"])

    scores = nx.pagerank(
        G,
        alpha=alpha,
        weight="weight",
        personalization=vetor_personalizacao,
        tol=1e-08
    )

    nome_col = f"PageRank_{coluna_peso or 'contagem'}"
    ranking = pd.DataFrame(list(scores.items()), columns=["Jogador", nome_col])
    return ranking.sort_values(by=nome_col, ascending=False).reset_index(drop=True)


def analisar_varias_partidas_selecao(
    lista_match_ids,
    time_alvo,
    coluna_peso=None,
    inverter_grafo=None,
    recorte_ate_sub=None,
    retornar_partidas=False
):
    """
    Calcula o PageRank jogo a jogo para uma lista de partidas e retorna a média
    de centralidade de cada jogador nos jogos em que participou.
    """
    if inverter_grafo is None:
        inverter_grafo = INVERTER_GRAFO
    if recorte_ate_sub is None:
        recorte_ate_sub = RECORTE_ATE_PRIMEIRA_SUBSTITUICAO

    todos_rankings = []
    nome_coluna_pr = f"PageRank_{coluna_peso or 'contagem'}"

    for match_id in lista_match_ids:
        ranking_jogo = calcular_pagerank_futebol(
            MATCH_ID=match_id,
            TIME_ALVO=time_alvo,
            coluna_peso=coluna_peso,
            inverter_grafo=inverter_grafo,
            recorte_ate_sub=recorte_ate_sub
        )
        ranking_jogo["match_id"] = match_id
        todos_rankings.append(ranking_jogo)

    df_consolidado = pd.concat(todos_rankings, ignore_index=True)

    df_media = df_consolidado.groupby("Jogador", as_index=False).agg(
        Jogos_Analisados=("match_id", "count"),
        Media_PageRank=(nome_coluna_pr, "mean")
    )

    df_media = df_media.sort_values(by="Media_PageRank", ascending=False).reset_index(drop=True)
    return (df_media, df_consolidado) if retornar_partidas else df_media


def executar_pipeline_completo(
    lista_match_ids=None,
    time_alvo=TIME_ALVO,
    inverter_grafo=None,
    recorte_ate_sub=None
):
    """
    Executa a calibração de parâmetros e avalia os três modelos do notebook
    corrigido ('Básico', 'xT' e 'xG híbrido') em todas as partidas selecionadas,
    retornando:
    - ranking_final: DataFrame consolidado com médias e posições nos 3 modelos
    - resumo_jogos: DataFrame de estabilidade e correlação de Spearman por partida
    - medias_por_modelo: dicionário {modelo: df_media}
    - rankings_partidas: dicionário {modelo: df_consolidado_partidas}
    """
    if lista_match_ids is None:
        lista_match_ids = IDS_JOGOS
    if inverter_grafo is None:
        inverter_grafo = INVERTER_GRAFO
    if recorte_ate_sub is None:
        recorte_ate_sub = RECORTE_ATE_PRIMEIRA_SUBSTITUICAO

    base_xg, fator_posse = calibrar_parametros_xg(
        lista_match_ids, time_alvo, recorte_ate_sub=recorte_ate_sub
    )
    print(f"[Calibração] Piso xT (mediana da grade): {CONSTANTE_BASE_XT:.8f}")
    print(f"[Calibração] Piso xG (mediana de chutes): {base_xg:.6f} | Fator xG posse: {fator_posse:.6f}")

    modelos = {"Básico": None, "xT": "peso_xt", "xG híbrido": "peso_xg"}
    medias = {}
    rankings_partidas = {}

    for modelo, coluna in modelos.items():
        medias[modelo], rankings_partidas[modelo] = analisar_varias_partidas_selecao(
            lista_match_ids=lista_match_ids,
            time_alvo=time_alvo,
            coluna_peso=coluna,
            inverter_grafo=inverter_grafo,
            recorte_ate_sub=recorte_ate_sub,
            retornar_partidas=True,
        )

    ranking_final = None
    for modelo, tabela in medias.items():
        tabela_renomeada = tabela.rename(columns={
            "Media_PageRank": f"PR_{modelo}",
            "Jogos_Analisados": f"Jogos_{modelo}",
        })
        ranking_final = (
            tabela_renomeada
            if ranking_final is None
            else ranking_final.merge(tabela_renomeada, on="Jogador", how="outer")
        )

    ranking_final = ranking_final.sort_values("PR_Básico", ascending=False).reset_index(drop=True)
    for modelo in modelos:
        ranking_final[f"Pos_{modelo}"] = (
            ranking_final[f"PR_{modelo}"]
            .rank(ascending=False, method="min")
            .astype("Int64")
        )

    comparacao_jogos = []
    for match_id in lista_match_ids:
        tabelas = {}
        for modelo, tabela in rankings_partidas.items():
            coluna = next(col for col in tabela.columns if col.startswith("PageRank_"))
            jogo = tabela.loc[tabela["match_id"] == match_id, ["Jogador", coluna]].copy()
            tabelas[modelo] = jogo.set_index("Jogador")[coluna].rename(modelo)

        alinhados = pd.concat(tabelas.values(), axis=1)
        posicoes = alinhados.rank(ascending=False, method="min")
        comparacao_jogos.append({
            "match_id": match_id,
            "Adversário": NOMES_JOGOS.get(match_id, str(match_id)),
            "Líder básico": alinhados["Básico"].idxmax(),
            "Líder xT": alinhados["xT"].idxmax(),
            "Líder xG": alinhados["xG híbrido"].idxmax(),
            "Mudanças xT": int((posicoes["Básico"] != posicoes["xT"]).sum()),
            "Mudanças xG": int((posicoes["Básico"] != posicoes["xG híbrido"]).sum()),
            "Spearman xT": float(spearmanr(alinhados["Básico"], alinhados["xT"]).statistic),
            "Spearman xG": float(spearmanr(alinhados["Básico"], alinhados["xG híbrido"]).statistic),
        })

    resumo_jogos = pd.DataFrame(comparacao_jogos)
    return ranking_final, resumo_jogos, medias, rankings_partidas


# ============================================================================
# CAMPO DE FUTEBOL (geometria StatsBomb)
# ============================================================================

def desenhar_campo_statsbomb(
    ax,
    cor_fundo=COR_FUNDO_CAMPO,
    cor_linha=COR_LINHAS_CAMPO,
    cor_seta_ataque="#334155",
    zorder_linhas=1
):
    """
    Renderiza o campo de futebol na orientação horizontal seguindo exatamente a
    geometria oficial da StatsBomb: X in [0, 120], Y in [0, 80] (Y invertido,
    com 0 no topo e 80 na base, ataque da esquerda para a direita em X=120, Y=40).
    Utiliza mplsoccer.Pitch quando instalado ou geometria vetorial equivalente.
    """
    if HAS_MPLSOCCER:
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color=cor_fundo,
            line_color=cor_linha,
            linewidth=1.25,
            goal_type="box",
            stripe=False,
            line_zorder=zorder_linhas
        )
        pitch.draw(ax=ax)
        return pitch

    ax.set_facecolor(cor_fundo)
    ax.set_xlim(-4, 124)
    ax.set_ylim(85, -4)
    ax.set_aspect("equal")
    ax.axis("off")

    lw = 1.25

    ax.add_patch(Rectangle((0, 0), 120, 80, fill=True, facecolor=cor_fundo,
                           edgecolor=cor_linha, lw=lw, zorder=0))
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False,
                           edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.plot([60, 60], [0, 80], color=cor_linha, lw=lw, zorder=zorder_linhas)

    ax.add_patch(Circle((60, 40), 10, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.scatter([60], [40], color=cor_linha, s=18, zorder=zorder_linhas + 0.5)

    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))

    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))

    ax.add_patch(Rectangle((-2.4, 36), 2.4, 8, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.add_patch(Rectangle((120, 36), 2.4, 8, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))

    ax.scatter([12, 108], [40, 40], color=cor_linha, s=15, zorder=zorder_linhas + 0.5)
    ax.add_patch(Arc((12, 40), 20, 20, angle=0, theta1=307, theta2=53, color=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.add_patch(Arc((108, 40), 20, 20, angle=0, theta1=127, theta2=233, color=cor_linha, lw=lw, zorder=zorder_linhas))

    ax.annotate(
        "Direção de Ataque ($x \\to 120$)",
        xy=(74, 82.8), xytext=(46, 82.8),
        ha="center", va="center", fontsize=9.5, color=cor_seta_ataque, style="italic",
        arrowprops=dict(arrowstyle="->", color=cor_seta_ataque, lw=1.25)
    )
    return None


# ============================================================================
# FUNÇÕES AUXILIARES DE ESTILO (textos dinâmicos derivados da configuração)
# ============================================================================

def _texto_alpha():
    """Descrição do fator de amortecimento conforme MODO_ALPHA."""
    if MODO_ALPHA == "fixo":
        return f"$\\alpha = {ALPHA_FIXO:.2f}$"
    return "$\\alpha$ empírico"


def _texto_recorte():
    """Descrição do recorte temporal conforme RECORTE_ATE_PRIMEIRA_SUBSTITUICAO."""
    if RECORTE_ATE_PRIMEIRA_SUBSTITUICAO:
        return "até a 1ª substituição"
    return "partida completa, prorrogação incluída e pênaltis excluídos"


def _estilo_tufte(ax, eixo_grade="x"):
    """Remove bordas superior/direita e aplica grade pontilhada discreta."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    if eixo_grade == "x":
        ax.xaxis.grid(True, linestyle=":", alpha=0.45, color="#94A3B8", zorder=0)
    else:
        ax.yaxis.grid(True, linestyle=":", alpha=0.45, color="#94A3B8", zorder=0)
    ax.set_axisbelow(True)


# ============================================================================
# VISUALIZAÇÃO 1 — Rede de passes + PageRank
# ============================================================================

def _separar_nos_sobrepostos(coords, raios, margem=5.0, limites=(4.0, 116.0, 5.0, 75.0),
                             n_iter_ancora=400, n_iter_livre=600, k_ancora=0.015):
    """
    Afasta iterativamente vértices que se sobrepõem (repulsão entre pares de
    círculos), mantendo-os o mais próximo possível da posição média tática.

    coords : array (n, 2) com as posições médias (x, y) em unidades do campo.
    raios  : array (n,) com o raio de cada vértice em unidades do campo.
    Fase 1 aplica uma mola fraca de volta à posição original; a fase 2 é
    repulsão pura, garantindo que ao final nenhum par de vértices se cruze.
    """
    pos = np.asarray(coords, dtype=float).copy()
    origem = pos.copy()
    n = len(pos)
    x_min, x_max, y_min, y_max = limites

    def _passo(usar_ancora):
        desloc = np.zeros_like(pos)
        houve = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                dist = float(np.hypot(d[0], d[1]))
                minimo = raios[i] + raios[j] + margem
                if dist < minimo:
                    houve = True
                    if dist < 1e-9:
                        ang = 2.0 * np.pi * (i * 7 + j * 3) / max(n, 1)
                        direcao = np.array([np.cos(ang), np.sin(ang)])
                    else:
                        direcao = d / dist
                    empurrao = (minimo - dist) / 2.0 + 1e-3
                    desloc[i] -= direcao * empurrao
                    desloc[j] += direcao * empurrao
        if usar_ancora:
            desloc += k_ancora * (origem - pos)
        pos[:] = pos + desloc
        pos[:, 0] = np.clip(pos[:, 0], x_min, x_max)
        pos[:, 1] = np.clip(pos[:, 1], y_min, y_max)
        return houve

    for _ in range(n_iter_ancora):
        _passo(usar_ancora=True)
    for _ in range(n_iter_livre):
        if not _passo(usar_ancora=False):
            break
    return pos


def _area_sobreposicao(a, b):
    """Área de interseção entre duas Bbox (em pixels de tela)."""
    largura = min(a.x1, b.x1) - max(a.x0, b.x0)
    altura = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(largura, 0.0) * max(altura, 0.0)


def _posicionar_rotulos_sem_colisao(fig, ax, nos_df, col_x, col_y, tamanhos_nos, col_pr):
    """
    Posiciona o rótulo de cada jogador testando candidatos ao redor do vértice
    (acima, abaixo, direita, esquerda e diagonais). Escolhe o primeiro candidato
    sem colisão com outros rótulos, outros vértices ou a borda da figura; se
    todos colidirem, usa o de menor área de sobreposição.
    """
    from matplotlib.transforms import Bbox

    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = fig._get_renderer()

    px_por_pt = fig.dpi / 72.0
    caixa_eixo = ax.get_window_extent(renderer)

    caixas_nos = []
    for k, (_, row) in enumerate(nos_df.iterrows()):
        cx, cy = ax.transData.transform((row[col_x], row[col_y]))
        r_px = (np.sqrt(tamanhos_nos.iloc[k]) / 2.0 + 1.5) * px_por_pt
        caixas_nos.append(Bbox([[cx - r_px, cy - r_px], [cx + r_px, cy + r_px]]))

    ordem = nos_df[col_pr].sort_values(ascending=False).index.tolist()
    caixas_rotulos = []

    for idx in ordem:
        k = nos_df.index.get_loc(idx)
        row = nos_df.loc[idx]
        r_pt = np.sqrt(tamanhos_nos.iloc[k]) / 2.0 + 3.0
        d = r_pt * 0.72
        candidatos = [
            (0, r_pt, "center", "bottom"),
            (0, -r_pt, "center", "top"),
            (r_pt, 0, "left", "center"),
            (-r_pt, 0, "right", "center"),
            (d, d, "left", "bottom"),
            (-d, d, "right", "bottom"),
            (d, -d, "left", "top"),
            (-d, -d, "right", "top"),
        ]
        texto = f"{row['nome_curto']}\n({row[col_pr]:.3f})"

        melhor, melhor_pen = None, np.inf
        for dx, dy, ha, va in candidatos:
            teste = ax.annotate(
                texto, xy=(row[col_x], row[col_y]), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va=va,
                fontsize=8.8, fontweight="bold", zorder=7
            )
            caixa = teste.get_window_extent(renderer)
            teste.remove()

            penalidade = sum(_area_sobreposicao(caixa, c) for c in caixas_rotulos)
            penalidade += sum(
                _area_sobreposicao(caixa, c) for j, c in enumerate(caixas_nos) if j != k
            )
            area_total = max(caixa.width * caixa.height, 1e-9)
            penalidade += 0.5 * (area_total - _area_sobreposicao(caixa, caixa_eixo))

            if penalidade < melhor_pen:
                melhor_pen = penalidade
                melhor = (dx, dy, ha, va, caixa)
            if penalidade == 0:
                break

        dx, dy, ha, va, caixa = melhor
        caixas_rotulos.append(caixa)
        txt = ax.annotate(
            texto, xy=(row[col_x], row[col_y]), xytext=(dx, dy),
            textcoords="offset points", ha=ha, va=va,
            fontsize=8.8, fontweight="bold", color=COR_TEXTO_ESCURO, zorder=7
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=3.0, foreground="#FFFFFF", alpha=0.94),
            path_effects.Normal()
        ])


def plotar_rede_passes_pagerank(
    df_passes,
    df_ranking,
    coluna_peso="peso_tatico",
    min_peso_aresta=2.5,
    min_passes_par=None,
    match_id=None,
    titulo=None,
    subtitulo=None,
    caminho_saida_pdf=f"{PASTA_FIGURAS}/fig_rede_passes.pdf",
    caminho_saida_png=f"{PASTA_FIGURAS}/fig_rede_passes.png"
):
    """
    Gera a visualização da Rede de Passes de UMA única partida sobre o gramado
    StatsBomb (df_passes e df_ranking devem ser da mesma partida, `match_id`):
    - Posição média (x, y) de cada jogador a partir de x_inicio e y_inicio.
    - Área e cor dos vértices proporcionais ao PageRank.
    - Arestas direcionadas curvas com espessura e opacidade proporcionais ao
      peso acumulado (peso_final) de cada par.

    Tratamento de sobreposição:
    - Vértices cujas posições médias se cruzam são afastados por repulsão
      (_separar_nos_sobrepostos), mantendo-os próximos da posição tática real.
    - Os rótulos são posicionados por busca de candidato sem colisão
      (_posicionar_rotulos_sem_colisao).
    - As setas terminam na borda de cada vértice (raio individual), e não sob ele.

    Filtros de aresta (combináveis):
    - min_peso_aresta: limiar absoluto sobre o peso acumulado do par (depende
      da escala da métrica).
    - min_passes_par: nº mínimo de passes do par (invariante à escala).
    """
    col_pr = [c for c in df_ranking.columns if "PageRank" in c or c.startswith("PR_")][0]

    posicoes_medias = (
        df_passes.groupby("player", as_index=False)
        .agg(
            x_medio=("x_inicio", "mean"),
            y_medio=("y_inicio", "mean"),
            passes_origem=("x_inicio", "count")
        )
        .rename(columns={"player": "Jogador"})
    )
    nos_df = pd.merge(
        posicoes_medias,
        df_ranking[["Jogador", col_pr]],
        on="Jogador",
        how="inner"
    ).reset_index(drop=True)
    nos_df["nome_curto"] = nos_df["Jogador"].map(abreviar_nome)

    if coluna_peso is None or coluna_peso not in df_passes.columns:
        arestas_df = (
            df_passes.groupby(["player", "pass_recipient"], as_index=False)
            .agg(contagem_passes=("x_inicio", "count"), peso_final=("x_inicio", "count"))
        )
        rotulo_peso = "Contagem de Passes ($w_{ij}$)"
    else:
        arestas_df = (
            df_passes.groupby(["player", "pass_recipient"], as_index=False)
            .agg(contagem_passes=(coluna_peso, "count"), peso_final=(coluna_peso, "sum"))
        )
        mapa_rotulos_peso = {
            "peso_xt": "Peso $xT$ ($w_{ij}^{xT}$)",
            "peso_xg": "Peso $xG$ Híbrido ($w_{ij}^{xG}$)",
            "peso_tatico": "Peso Tático ($w_{ij}$)"
        }
        rotulo_peso = mapa_rotulos_peso.get(coluna_peso, f"Peso ({coluna_peso})")

    if min_passes_par is not None:
        arestas_df = arestas_df[arestas_df["contagem_passes"] >= min_passes_par]
    if min_peso_aresta is not None:
        arestas_df = arestas_df[arestas_df["peso_final"] >= min_peso_aresta]

    arestas_df = arestas_df[
        arestas_df["player"].isin(nos_df["Jogador"])
        & arestas_df["pass_recipient"].isin(nos_df["Jogador"])
    ].copy()

    # ---------------- Figura, campo e barra de cores (definem a geometria) -------------
    fig, ax = plt.subplots(figsize=(12, 8))
    desenhar_campo_statsbomb(ax)

    pr_min = nos_df[col_pr].min()
    pr_max = nos_df[col_pr].max()
    norm_pr = Normalize(vmin=pr_min * 0.90, vmax=pr_max * 1.02)
    cmap_nos = LinearSegmentedColormap.from_list(
        "argentina_pr", ["#BAE6FD", COR_CELESTE_ARG, COR_AZUL_PROFUNDO]
    )
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_pr, cmap=cmap_nos),
        ax=ax, orientation="horizontal", fraction=0.038, pad=0.03, aspect=38
    )
    cbar.set_label(
        "Centralidade Espectral de PageRank ($\\pi_i$)",
        fontsize=10.5, fontweight="bold", labelpad=6
    )
    cbar.ax.tick_params(labelsize=9)

    # Com a geometria final dos eixos, converte pontos -> unidades do campo
    ax.apply_aspect()
    fig.canvas.draw()
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 0.0))
    pt_por_unidade = abs(p1[0] - p0[0]) * 72.0 / fig.dpi

    # ---------------- Tamanhos e posições dos vértices (sem sobreposição) -------------
    tamanhos_nos = 380 + 1070 * ((nos_df[col_pr] - pr_min) / max(pr_max - pr_min, 1e-9))
    raios_pt = np.sqrt(tamanhos_nos.to_numpy()) / 2.0 + 1.1  # inclui a borda branca
    raios_unid = raios_pt / pt_por_unidade

    coords_originais = nos_df[["x_medio", "y_medio"]].to_numpy()
    coords_finais = _separar_nos_sobrepostos(coords_originais, raios_unid)
    nos_df["x_plot"] = coords_finais[:, 0]
    nos_df["y_plot"] = coords_finais[:, 1]
    deslocamento_max = float(np.hypot(*(coords_finais - coords_originais).T).max())

    # ---------------- Arestas ----------------
    if arestas_df.empty:
        print("[AVISO] Nenhuma aresta atende aos limiares informados; "
              "a rede será desenhada apenas com os vértices.")
    else:
        pos = nos_df.set_index("Jogador")[["x_plot", "y_plot"]]
        raio_pt_por_jogador = pd.Series(raios_pt, index=nos_df["Jogador"])

        peso_min = arestas_df["peso_final"].min()
        peso_max = arestas_df["peso_final"].max()
        amplitude_peso = max(peso_max - peso_min, 1e-9)
        arestas_df = arestas_df.sort_values(by="peso_final", ascending=True)

        for _, row in arestas_df.iterrows():
            fator_norm = (row["peso_final"] - peso_min) / amplitude_peso
            lw_aresta = 1.0 + 5.2 * fator_norm
            alpha_aresta = 0.22 + 0.68 * fator_norm
            origem, destino = row["player"], row["pass_recipient"]

            ax.annotate(
                "",
                xy=tuple(pos.loc[destino]),
                xytext=tuple(pos.loc[origem]),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.65,head_width=0.45",
                    color=COR_AZUL_PROFUNDO,
                    lw=lw_aresta,
                    alpha=alpha_aresta,
                    shrinkA=raio_pt_por_jogador[origem] + 1.0,
                    shrinkB=raio_pt_por_jogador[destino] + 2.0,
                    connectionstyle="arc3,rad=0.12"
                ),
                zorder=3
            )

    # ---------------- Vértices ----------------
    ax.scatter(
        nos_df["x_plot"], nos_df["y_plot"],
        s=tamanhos_nos, c=nos_df[col_pr], cmap=cmap_nos, norm=norm_pr,
        edgecolors="#FFFFFF", linewidths=2.2, zorder=5
    )
    ax.scatter(
        nos_df["x_plot"], nos_df["y_plot"],
        s=tamanhos_nos, facecolors="none",
        edgecolors=COR_TEXTO_ESCURO, linewidths=0.9, zorder=6
    )

    # ---------------- Rótulos sem colisão ----------------
    _posicionar_rotulos_sem_colisao(fig, ax, nos_df, "x_plot", "y_plot", tamanhos_nos, col_pr)

    # ---------------- Títulos ----------------
    adversario = NOMES_JOGOS.get(match_id, None)
    if titulo is None:
        if adversario:
            titulo = (f"Rede Direcionada de Passes e Centralidade PageRank — "
                      f"{TIME_ALVO} × {adversario} (Copa 2022)")
        else:
            titulo = "Rede Direcionada de Passes e Centralidade PageRank — Seleção Argentina (Copa 2022)"

    if subtitulo is None:
        id_partida = f"Partida {match_id}" if match_id is not None else "Partida"
        if adversario:
            id_partida += f" (vs. {adversario})"
        ajuste = (" | posições médias levemente ajustadas para evitar sobreposição"
                  if deslocamento_max > 0.5 else "")
        subtitulo = (
            f"{id_partida}, {_texto_recorte()} | {_texto_alpha()}\n"
            f"Área do Nó $\\propto$ PageRank ($\\pi_i$) | Espessura $\\propto$ {rotulo_peso}{ajuste}"
        )

    ax.set_title(titulo, fontsize=13.5, fontweight="bold", pad=38, color=COR_TEXTO_ESCURO, loc="left")
    ax.text(0, -5.8, subtitulo, fontsize=9.8, color=COR_LINHAS_CAMPO, style="italic", va="bottom", linespacing=1.35)

    _garantir_pasta(caminho_saida_pdf, caminho_saida_png)
    plt.savefig(caminho_saida_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(caminho_saida_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Visualização 1 (Rede de Passes) salva em: {caminho_saida_pdf} e {caminho_saida_png}")


# ============================================================================
# VISUALIZAÇÃO 2 — Barras agrupadas: PageRank ponderado por xT vs. xG híbrido
# ============================================================================

def plotar_barras_agrupadas_xt_xg(
    ranking_final,
    top_n=14,
    titulo=None,
    subtitulo=None,
    caminho_saida_pdf=f"{PASTA_FIGURAS}/barras_agrupadas_xt_xg.pdf",
    caminho_saida_png=f"{PASTA_FIGURAS}/barras_agrupadas_xt_xg.png"
):
    """
    Barras agrupadas (verticais) comparando a média de PageRank dos `top_n`
    jogadores (ordenados pelo modelo xT) sob os três modelos da análise:
    Básico, xT e xG híbrido.

    Mantém a mesma lógica visual do gráfico original do colega, acrescentando
    apenas a terceira barra referente ao PageRank básico.
    """
    colunas_necessarias = ["PR_Básico", "PR_xT", "PR_xG híbrido"]
    df_plot = (
        ranking_final.dropna(subset=colunas_necessarias)
        .sort_values("PR_xT", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    n_jogadores = len(df_plot)
    n_partidas = int(ranking_final[[c for c in ranking_final.columns if c.startswith("Jogos_")]].max().max())
    x = np.arange(n_jogadores)
    largura = 0.24

    coluna_jogos_rotulo = "Jogos_Básico" if "Jogos_Básico" in df_plot.columns else "Jogos_xT"
    rotulos_x = [
        f"{abreviar_nome(r['Jogador'])}\n({int(r[coluna_jogos_rotulo])}J)"
        for _, r in df_plot.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(26, 7.6))

    barras_basico = ax.bar(
        x - largura, df_plot["PR_Básico"], largura,
        color=COR_CINZA_TUFTE, edgecolor=COR_TEXTO_ESCURO, linewidth=0.6,
        label="PageRank básico", zorder=3
    )
    barras_xt = ax.bar(
        x, df_plot["PR_xT"], largura,
        color=COR_CELESTE_ARG, edgecolor=COR_TEXTO_ESCURO, linewidth=0.6,
        label="Ponderação xT (Expected Threat)", zorder=3
    )
    barras_xg = ax.bar(
        x + largura, df_plot["PR_xG híbrido"], largura,
        color=COR_AZUL_PROFUNDO, edgecolor=COR_TEXTO_ESCURO, linewidth=0.6,
        label="Ponderação xG híbrido (Expected Goals)", zorder=3
    )

    max_y = float(max(
        df_plot["PR_Básico"].max(),
        df_plot["PR_xT"].max(),
        df_plot["PR_xG híbrido"].max()
    ))
    for barras in (barras_basico, barras_xt, barras_xg):
        for barra in barras:
            ax.text(
                barra.get_x() + barra.get_width() / 2.0,
                barra.get_height() + max_y * 0.006,
                f"{barra.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.3, color=COR_TEXTO_ESCURO, zorder=5
            )

    ax.set_xticks(x, rotulos_x)
    ax.set_xlim(-1.12, n_jogadores - 1 + 1.12)
    ax.set_ylim(0, max_y * 1.12)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9.2)
    ax.set_ylabel(r"Média de Centralidade PageRank ($\bar{\pi}_i$)", fontsize=11, fontweight="bold", labelpad=10)
    _estilo_tufte(ax, eixo_grade="y")

    tipo_pr = "do Fornecedor" if INVERTER_GRAFO else "do Recebedor"
    if titulo is None:
        titulo = f"PageRank {tipo_pr}: Básico vs. xT vs. xG Híbrido — Seleção Argentina (Copa 2022)"
    if subtitulo is None:
        subtitulo = (
            f"Média nas {n_partidas} partidas analisadas | Top {n_jogadores} jogadores por xT | "
            f"{_texto_alpha()}, {_texto_recorte()}"
        )

    ax.set_title(titulo, fontsize=13.5, fontweight="bold", pad=26, loc="left", color=COR_TEXTO_ESCURO)
    ax.text(
        0.0, 1.015, subtitulo, transform=ax.transAxes,
        fontsize=10, color=COR_LINHAS_CAMPO, style="italic", va="bottom", ha="left"
    )
    ax.legend(loc="upper right", frameon=True, facecolor=COR_FUNDO_CAMPO, edgecolor="#CBD5E1", fontsize=10.5)

    _garantir_pasta(caminho_saida_pdf, caminho_saida_png)
    plt.savefig(caminho_saida_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(caminho_saida_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Visualização 2 (Barras Agrupadas: Básico vs. xT vs. xG) salva em: {caminho_saida_pdf} e {caminho_saida_png}")


# ============================================================================
# VISUALIZAÇÃO 3 — Mapa de calor de Expected Threat (xT)
# ============================================================================

def plotar_mapa_calor_xt(
    xt_grid,
    df_passes=None,
    top_n_passes=12,
    titulo="Superfície Espacial de Expected Threat (xT) — Grade $8 \\times 12$ de Karun Singh",
    subtitulo="Probabilidade empírica de conversão em gol por zona do gramado StatsBomb ($120 \\times 80$) e vetores dos passes de maior $\\Delta xT$",
    caminho_saida_pdf=f"{PASTA_FIGURAS}/fig_heatmap_xt.pdf",
    caminho_saida_png=f"{PASTA_FIGURAS}/fig_heatmap_xt.png"
):
    """
    Gera o Mapa de Calor sobre o campo de futebol (120x80):
    - Sobrepõe a matriz 8x12 de Expected Threat (xT) usando colormap perceptualmente
      uniforme ('magma') e anota o valor exato de xT em cada uma das 96 zonas.
    - Caso df_passes seja fornecido, plota os vetores dos `top_n_passes` passes de
      maior geração de ameaça territorial (Delta xT) da partida.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    desenhar_campo_statsbomb(
        ax,
        cor_fundo="#18181B",
        cor_linha="#E2E8F0",
        cor_seta_ataque="#334155",
        zorder_linhas=2.2
    )

    x_edges = np.linspace(0, 120, 13)
    y_edges = np.linspace(0, 80, 9)

    mesh = ax.pcolormesh(
        x_edges, y_edges, xt_grid,
        cmap="magma", alpha=0.90, edgecolors="#FFFFFF", linewidth=0.35, zorder=1.5
    )

    for r in range(8):
        for c in range(12):
            val = xt_grid[r, c]
            x_centro = c * 10.0 + 5.0
            y_centro = r * 10.0 + 5.0
            cor_texto = COR_TEXTO_ESCURO if val > 0.08 else COR_FUNDO_CAMPO
            peso_fonte = "bold" if val > 0.035 else "normal"
            ax.text(
                x_centro, y_centro, f"{val:.3f}",
                ha="center", va="center", fontsize=7.5,
                color=cor_texto, fontweight=peso_fonte, zorder=3
            )

    if df_passes is not None and "delta_xt" in df_passes.columns:
        top_xt_passes = df_passes.sort_values(by="delta_xt", ascending=False).head(top_n_passes)
        for _, row in top_xt_passes.iterrows():
            ax.annotate(
                "",
                xy=(row["x_fim"], row["y_fim"]),
                xytext=(row["x_inicio"], row["y_inicio"]),
                arrowprops=dict(
                    arrowstyle="->,head_length=0.55,head_width=0.35",
                    color="#38BDF8",
                    lw=1.8,
                    alpha=0.92
                ),
                zorder=4
            )

    cbar = fig.colorbar(
        mesh, ax=ax, orientation="horizontal",
        fraction=0.038, pad=0.03, aspect=38
    )
    cbar.set_label("Valor Posicional de Expected Threat ($xT_{r,c}$)", fontsize=10.5, fontweight="bold", labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(titulo, fontsize=13.5, fontweight="bold", pad=22, color=COR_TEXTO_ESCURO, loc="left")
    ax.text(0, -5.8, subtitulo, fontsize=9.8, color=COR_LINHAS_CAMPO, style="italic")

    _garantir_pasta(caminho_saida_pdf, caminho_saida_png)
    plt.savefig(caminho_saida_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(caminho_saida_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Visualização 3 (Mapa de Calor xT) salva em: {caminho_saida_pdf} e {caminho_saida_png}")

import os
import sys


def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")


@lru_cache(maxsize=1)
def carregar_partidas_copa_2022():
    """Carrega a tabela de partidas da Copa do Mundo 2022 via StatsBomb."""
    print("Carregando base de partidas da Copa do Mundo 2022 (StatsBomb)...")
    return sb.matches(competition_id=43, season_id=106)


def escolher_opcao(opcoes, prompt="Escolha uma opção: "):
    """Lê e valida um índice numérico escolhido pelo usuário."""
    while True:
        entrada = input(prompt).strip()
        if entrada.lower() in ("q", "sair"):
            return None
        if entrada.isdigit():
            idx = int(entrada)
            if 0 <= idx < len(opcoes):
                return idx
        print(f"Opção inválida. Digite um número entre 0 e {len(opcoes) - 1} (ou 'q' para sair).")


def selecionar_selecao(df_partidas):
    """Lista as 32 seleções disponíveis e retorna o nome da escolhida."""
    selecoes = sorted(set(df_partidas["home_team"]).union(set(df_partidas["away_team"])))
    print("\n=== SELEÇÕES DISPONÍVEIS (COPA 2022) ===")
    for i, sel in enumerate(selecoes):
        print(f"  [{i:2d}] {sel}")
    print("  [ q] Sair")

    idx = escolher_opcao(selecoes, "\nDigite o número da seleção: ")
    return None if idx is None else selecoes[idx]


def selecionar_jogo(df_partidas, selecao):
    """Lista os jogos da seleção escolhida e permite escolher um jogo ou todos."""
    jogos = df_partidas[
        (df_partidas["home_team"] == selecao) | (df_partidas["away_team"] == selecao)
    ].sort_values("match_date").reset_index(drop=True)

    mapa_adversarios = {}
    opcoes_texto = ["TODOS OS JOGOS (Campanha Completa)"]

    for _, row in jogos.iterrows():
        mid = int(row["match_id"])
        adversario = row["away_team"] if row["home_team"] == selecao else row["home_team"]
        placar = f"{row['home_team']} {row['home_score']} x {row['away_score']} {row['away_team']}"
        fase = row.get("competition_stage", "")
        mapa_adversarios[mid] = adversario
        opcoes_texto.append(f"{placar}  ({fase}) [ID: {mid}]")

    print(f"\n=== PARTIDAS DE {selecao.upper()} ===")
    for i, txt in enumerate(opcoes_texto):
        print(f"  [{i}] {txt}")

    idx = escolher_opcao(opcoes_texto, "\nEscolha a partida (ou 0 para campanha completa): ")
    if idx is None:
        return None, None, mapa_adversarios

    if idx == 0:
        ids_escolhidos = jogos["match_id"].astype(int).tolist()
        return "campanha", ids_escolhidos, mapa_adversarios

    match_id = int(jogos.loc[idx - 1, "match_id"])
    return "jogo_unico", [match_id], mapa_adversarios


def selecionar_peso_rede():
    """Permite escolher a ponderação das arestas para a rede de passes."""
    opcoes = [
        ("peso_xt", "Expected Threat (xT)"),
        ("peso_xg", "Expected Goals Híbrido (xG)"),
        ("peso_tatico", "Peso Tático"),
        (None, "Básico (Contagem de Passes)"),
    ]
    print("\n=== MODELO DE PONDERAÇÃO PARA A REDE DE PASSES ===")
    for i, (_, desc) in enumerate(opcoes):
        print(f"  [{i}] {desc}")
    idx = escolher_opcao(opcoes, "\nEscolha o modelo: ")
    return "peso_xt" if idx is None else opcoes[idx][0]


def rodar_cli():
    global TIME_ALVO, NOMES_JOGOS

    Path(PASTA_FIGURAS).mkdir(exist_ok=True)
    Path("resultados").mkdir(exist_ok=True)

    df_partidas = carregar_partidas_copa_2022()

    while True:
        selecao = selecionar_selecao(df_partidas)
        if selecao is None:
            print("Encerrando.")
            break

        modo, ids_escolhidos, mapa_adversarios = selecionar_jogo(df_partidas, selecao)
        if modo is None:
            continue

        # Atualiza as globais usadas nos títulos e tabelas do pipeline original
        TIME_ALVO = selecao
        NOMES_JOGOS = mapa_adversarios

        gerar_figs = input("\nDeseja gerar e salvar os gráficos em PDF/PNG? (s/n) [s]: ").strip().lower()
        salvar_graficos = gerar_figs in ("", "s", "sim")

        print(f"\nProcessando análise para {selecao}...")

        # Executa o pipeline completo (funciona tanto para 1 jogo quanto para a campanha inteira)
        ranking_final, resumo_jogos, _, _ = executar_pipeline_completo(
            lista_match_ids=ids_escolhidos,
            time_alvo=selecao,
            inverter_grafo=INVERTER_GRAFO,
            recorte_ate_sub=RECORTE_ATE_PRIMEIRA_SUBSTITUICAO,
        )

        slug_time = selecao.lower().replace(" ", "_")

        if modo == "campanha":
            print(f"\n--- Ranking Médio Consolidado ({len(ids_escolhidos)} Jogos — {selecao}) ---")
            print(ranking_final.to_string(index=False))

            print(f"\n--- Estabilidade e Correlação de Spearman por Partida ({selecao}) ---")
            print(resumo_jogos.to_string(index=False))

            ranking_final.to_csv(f"resultados/ranking_medio_{slug_time}.csv", index=False)
            resumo_jogos.to_csv(f"resultados/comparacao_partidas_{slug_time}.csv", index=False)

            if salvar_graficos:
                tipo_pr = "do Fornecedor" if INVERTER_GRAFO else "do Recebedor"
                plotar_barras_agrupadas_xt_xg(
                    ranking_final=ranking_final,
                    top_n=14,
                    titulo=f"PageRank {tipo_pr}: Básico vs. xT vs. xG Híbrido — {selecao} (Copa 2022)",
                    caminho_saida_pdf=f"{PASTA_FIGURAS}/barras_{slug_time}.pdf",
                    caminho_saida_png=f"{PASTA_FIGURAS}/barras_{slug_time}.png",
                )

        else:
            match_id = ids_escolhidos[0]
            adversario = mapa_adversarios.get(match_id, str(match_id))
            slug_adv = adversario.lower().replace(" ", "_")

            print(f"\n--- Ranking da Partida: {selecao} x {adversario} (ID {match_id}) ---")
            print(ranking_final.to_string(index=False))

            print("\n--- Estabilidade e Correlação de Spearman ---")
            print(resumo_jogos.to_string(index=False))

            ranking_final.to_csv(f"resultados/ranking_{slug_time}_vs_{slug_adv}.csv", index=False)

            if salvar_graficos:
                coluna_peso = selecionar_peso_rede()
                df_passes = obter_df_passes(
                    match_id, selecao, recorte_ate_sub=RECORTE_ATE_PRIMEIRA_SUBSTITUICAO
                )
                df_ranking_rede = calcular_pagerank_futebol(
                    match_id,
                    selecao,
                    coluna_peso=coluna_peso,
                    inverter_grafo=INVERTER_GRAFO,
                    recorte_ate_sub=RECORTE_ATE_PRIMEIRA_SUBSTITUICAO,
                )

                plotar_rede_passes_pagerank(
                    df_passes=df_passes,
                    df_ranking=df_ranking_rede,
                    coluna_peso=coluna_peso,
                    min_passes_par=3,
                    min_peso_aresta=None,
                    match_id=match_id,
                    caminho_saida_pdf=f"{PASTA_FIGURAS}/rede_{slug_time}_vs_{slug_adv}.pdf",
                    caminho_saida_png=f"{PASTA_FIGURAS}/rede_{slug_time}_vs_{slug_adv}.png",
                )

                plotar_mapa_calor_xt(
                    xt_grid=XT_GRID,
                    df_passes=df_passes,
                    top_n_passes=15,
                    subtitulo=(
                        "Probabilidade empírica de conversão em gol por zona do gramado StatsBomb "
                        f"($120 \\times 80$) e os 15 passes de maior $\\Delta xT$ em {selecao} vs. {adversario}"
                    ),
                    caminho_saida_pdf=f"{PASTA_FIGURAS}/heatmap_xt_{slug_time}_vs_{slug_adv}.pdf",
                    caminho_saida_png=f"{PASTA_FIGURAS}/heatmap_xt_{slug_time}_vs_{slug_adv}.png",
                )

        continuar = input("\nDeseja fazer outra análise? (s/n) [s]: ").strip().lower()
        if continuar not in ("", "s", "sim"):
            print("Encerrando.")
            break
        limpar_tela()


if __name__ == "__main__":
    try:
        rodar_cli()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(0)