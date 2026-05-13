import { createAboutMarkdownResponse } from '@/lib/site-content';
import { rootLocale } from '@/lib/route-helpers';

export function GET() {
  return createAboutMarkdownResponse(rootLocale);
}
