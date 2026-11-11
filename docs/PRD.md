# PRD Técnico — FraudGuard AI

**Versão:** 1.0.0  
**Estado:** Aprovado como especificação inicial; em implementação  
**Produto:** Sistema Inteligente de Detecção de Fraudes  
**Contexto:** Mini Capstone — Pós-graduação em Engenharia de Inteligência Artificial / Fundamentos de IA  
**Data:** 11/11/2026  
**Stack principal:** Python, FastAPI, Scikit-learn, Streamlit, PostgreSQL (Supabase), SQLAlchemy, Alembic, Docker Compose, GitHub Actions  
**Ferramentas de desenvolvimento:** Google Antigravity (IDE principal) e OpenAI Codex (agente de implementação/revisão)

---

> Este documento é a cópia canônica de `docs/PRD_FraudGuard_AI.md` conforme especificado na Seção 12.1 e Seção 18.1 do PRD.

## 1. Resumo executivo

O **FraudGuard AI** é uma aplicação demonstrativa de detecção de fraudes em transações financeiras. Utiliza classificação supervisionada, recebe transações pela API REST, calcula escores de risco, persiste previsões e disponibiliza um dashboard com indicadores, histórico e alertas para revisão humana.

O sistema trabalhará com o arquivo fornecido **`creditcard_2023.csv`**, contendo transações anonimizadas. O produto é **acadêmico e demonstrativo**, não deve ser interpretado como um sistema bancário certificado, nem deve autorizar, bloquear ou liquidar pagamentos reais.

### 1.1 Problema

Identificar transações potencialmente fraudulentas em grandes volumes exige métodos reproduzíveis, rastreáveis e avaliáveis. O projeto demonstra o ciclo completo: ingestão e validação de dados, treinamento e avaliação, publicação de um modelo, inferência via API, persistência e monitoramento básico.

### 1.2 Objetivos

1. Construir e comparar **Regressão Logística** (baseline) e **Random Forest**.
2. Disponibilizar um endpoint de classificação com predição reprodutível e metadados de versão.
3. Persistir transações, previsões, modelos, importações e alertas no PostgreSQL gerenciado pelo Supabase.
4. Exibir informações gerenciais e métricas de ML em um dashboard Streamlit.
5. Entregar código testado, documentado e reproduzível em um monorepositório GitHub.

### 1.3 Critérios de sucesso do MVP

- Importar e validar o CSV sem perder registros, com contagem por classe reconciliada.
- Treinar dois pipelines reproduzíveis, com avaliações registradas em um conjunto de teste intocado.
- Disponibilizar API com autenticação de operação administrativa, validação de schema e Swagger.
- Para a mesma versão do modelo e os mesmos atributos, retornar a mesma predição.
- Persistir a referência do modelo, pontuação, limiar e momento de cada inferência.
- Apresentar dashboard com transações, previsões, alertas e métricas reais dos experimentos.
- Passar nos testes automatizados e permitir inicialização documentada com Docker Compose.

**Sem meta arbitrária de 95% de acurácia.** Os resultados devem ser reportados honestamente, com matriz de confusão, Precision, Recall, F1, ROC-AUC e Average Precision, e comparados a uma baseline.

