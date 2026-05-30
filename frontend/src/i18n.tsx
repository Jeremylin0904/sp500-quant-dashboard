import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

export type Lang = "en" | "zh";

type Ctx = {
  lang: Lang;
  setLang: (l: Lang) => void;
  /** Pick a string by language. zh first arg, en second. English is the default. */
  t: (zh: string, en: string) => string;
};

const LangContext = createContext<Ctx>({
  lang: "en",
  setLang: () => {},
  t: (_zh, en) => en,
});

export function useLang(): Ctx {
  return useContext(LangContext);
}

/** Convenience hook returning just the translate function. */
export function useT(): (zh: string, en: string) => string {
  return useContext(LangContext).t;
}

export function readInitialLang(): Lang {
  try {
    const q = new URLSearchParams(window.location.search).get("lang");
    if (q === "zh" || q === "en") return q;
    const s = localStorage.getItem("lang");
    if (s === "zh" || s === "en") return s as Lang;
  } catch {
    /* ignore */
  }
  return "en"; // English by default
}

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(readInitialLang);

  const setLang = (l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem("lang", l);
      const u = new URL(window.location.href);
      u.searchParams.set("lang", l);
      // Keep the hash (current tab) intact.
      window.history.replaceState(null, "", u.toString());
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    document.documentElement.lang = lang === "zh" ? "zh-Hant" : "en";
    document.title =
      lang === "zh" ? "S&P 500 量化選股 Dashboard" : "S&P 500 Quant Stock-Selection Dashboard";
  }, [lang]);

  const t = (zh: string, en: string) => (lang === "zh" ? zh : en);

  return (
    <LangContext.Provider value={{ lang, setLang, t }}>{children}</LangContext.Provider>
  );
}
