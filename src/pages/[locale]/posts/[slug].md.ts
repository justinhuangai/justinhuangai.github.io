import {
  nonDefaultLocales,
  type Locale,
} from '@/i18n';
import {
  createMarkdownResponse,
  createMarkdownStaticPaths,
} from '@/lib/markdown-endpoint';

export async function getStaticPaths() {
  const paths: Array<{
    params: { locale: string; slug: string };
    props: { locale: Locale; markdownBody: string };
  }> = [];
  for (const locale of nonDefaultLocales) {
    const localePaths = await createMarkdownStaticPaths(locale)();
    localePaths.forEach((path) => {
      paths.push({
        params: { locale, ...path.params },
        props: { locale, ...path.props },
      });
    });
  }
  return paths;
}

export function GET({ props }: { props: { locale: Locale; markdownBody: string } }) {
  return createMarkdownResponse(props.markdownBody);
}
