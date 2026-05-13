export declare const AITHER_DEFAULT_LOCALE: 'en-US';

export declare const AITHER_LOCALES: readonly [
  'en-US',
  'zh-CN',
  'zh-TW',
  'ko-KR',
  'fr-FR',
  'de-DE',
  'it-IT',
  'es-ES',
  'ru-RU',
  'id-ID',
  'pt-BR',
];

export type AitherLocale = (typeof AITHER_LOCALES)[number];

export declare const AITHER_NODE_RANGE: '^20.19.1 || >=22.12.0';

export declare const AITHER_LOCALE_LABELS: Record<
  (typeof AITHER_LOCALES)[number],
  string
>;

export declare const AITHER_INTL_LOCALES: Record<
  (typeof AITHER_LOCALES)[number],
  string
>;

export declare const AITHER_HTML_LANGS: Record<
  (typeof AITHER_LOCALES)[number],
  string
>;

export declare const AITHER_CRISP_LOCALES: Record<
  (typeof AITHER_LOCALES)[number],
  string
>;

export declare const AITHER_GISCUS_LOCALES: Record<
  (typeof AITHER_LOCALES)[number],
  string
>;
