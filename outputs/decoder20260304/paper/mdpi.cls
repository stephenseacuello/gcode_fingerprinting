%%
%% mdpi.cls — Minimal MDPI journal class for local compilation
%% Based on the MDPI LaTeX template structure (https://www.mdpi.com/authors/latex)
%% Replace with the official mdpi.cls from MDPI before final submission.
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mdpi}[2026/03/22 Minimal MDPI journal class]

%% ---- Option handling ----
\newif\if@submit\@submitfalse
\newif\if@pdftex\@pdftexfalse
\newif\if@moreauthors\@moreauthorsfalse
\DeclareOption{submit}{\@submittrue}
\DeclareOption{pdftex}{\@pdftextrue}
\DeclareOption{moreauthors}{\@moreauthorstrue}
% Journal name option (sensors, machines, etc.) — store but ignore for formatting
\def\@journal{sensors}
\DeclareOption{sensors}{\def\@journal{sensors}}
\DeclareOption{machines}{\def\@journal{machines}}
\DeclareOption{article}{}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

%% ---- Base class ----
\LoadClass[10pt]{article}

%% ---- Required packages ----
\RequirePackage[T1]{fontenc}
\RequirePackage[margin=1in]{geometry}
\RequirePackage{mathpazo}          % Palatino (MDPI uses similar serif)
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{xcolor}
\RequirePackage{float}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{etoolbox}
\RequirePackage{lastpage}

%% ---- MDPI colors ----
\definecolor{mdpiblue}{RGB}{0,68,136}

%% ---- Page layout (single column, MDPI style) ----
\geometry{
  a4paper,
  left=25.4mm,
  right=25.4mm,
  top=25.4mm,
  bottom=25.4mm,
  columnsep=8mm
}

%% ---- Header/footer ----
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\thepage}

%% ---- Section formatting (MDPI numbered, bold) ----
\titleformat{\section}{\normalfont\large\bfseries}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries\itshape}{\thesubsection.}{0.5em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\itshape}{\thesubsubsection.}{0.5em}{}
\titlespacing*{\section}{0pt}{12pt plus 2pt minus 2pt}{6pt plus 1pt minus 1pt}
\titlespacing*{\subsection}{0pt}{8pt plus 2pt minus 2pt}{4pt plus 1pt minus 1pt}

%% ---- MDPI author/affiliation commands ----
% \Author{First Last \orcid{xxxx-xxxx-xxxx-xxxx}}
% \address{1}{affiliation text}
% \corres{email}
\newcommand{\orcid}[1]{\textsuperscript{\scriptsize\textcolor{mdpiblue}{#1}}}
\newcommand{\Author}[1]{\def\@mdpiauthors{#1}}
\newcommand{\address}[2]{%
  \@ifundefined{@mdpiaffiliations}{%
    \def\@mdpiaffiliations{\textsuperscript{#1}~#2}%
  }{%
    \g@addto@macro\@mdpiaffiliations{\\[2pt]\textsuperscript{#1}~#2}%
  }%
}
\newcommand{\corres}[1]{\def\@mdpicorres{#1}}
\newcommand{\firstnote}[1]{} % placeholder
\newcommand{\secondnote}[1]{} % placeholder

%% ---- Abstract and Keywords (MDPI style: commands, not environments) ----
\newcommand{\Abstract}[1]{\def\@mdpiabstract{#1}}
\renewcommand{\abstract}[1]{\def\@mdpiabstract{#1}}
\newcommand{\keyword}[1]{\def\@mdpikeywords{#1}}

%% ---- Title page rendering ----
\renewcommand{\maketitle}{%
  \begingroup
  \centering
  {\Large\bfseries \@title \par}
  \vskip 12pt
  {\normalsize \@mdpiauthors \par}
  \vskip 6pt
  {\small\itshape \@mdpiaffiliations \par}
  \vskip 4pt
  {\small Correspondence: \@mdpicorres \par}
  \vskip 12pt
  \endgroup
  %% Abstract
  \noindent\textbf{Abstract:}~\@mdpiabstract
  \vskip 8pt
  %% Keywords
  \noindent\textbf{Keywords:}~\@mdpikeywords
  \vskip 16pt
}

%% ---- \reftitle command (MDPI requires this before bibliography) ----
\newcommand{\reftitle}[1]{\section*{#1}}

%% ---- Abbreviations environment ----
\newenvironment{abbreviations}{%
  \section*{Abbreviations}
  \noindent
}{%
  \par\vskip 8pt
}

%% ---- Compatibility ----
% \abstract is already redefined above as a command via \renewcommand

%% ---- Journal name accessor ----
\newcommand{\journalname}{\@journal}

\endinput
