export const localeMeta = /** @type {const} */ ([
  {
    code: 'en-US',
    label: 'English',
    intl: 'en-US',
    htmlLang: 'en-US',
    crispLocale: 'en',
    giscusLocale: 'en',
  },
  {
    code: 'zh-CN',
    label: '简体中文',
    intl: 'zh-CN',
    htmlLang: 'zh-CN',
    crispLocale: 'zh',
    giscusLocale: 'zh-CN',
  },
  {
    code: 'zh-TW',
    label: '繁體中文',
    intl: 'zh-TW',
    htmlLang: 'zh-TW',
    crispLocale: 'zh',
    giscusLocale: 'zh-TW',
  },
  {
    code: 'ko-KR',
    label: '한국어',
    intl: 'ko-KR',
    htmlLang: 'ko-KR',
    crispLocale: 'ko',
    giscusLocale: 'ko',
  },
]);

export const defaultLocale = 'en-US';
export const locales = localeMeta.map((entry) => entry.code);
