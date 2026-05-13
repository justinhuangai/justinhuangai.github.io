import { createRssResponse } from '@/lib/site-content';
import { rootLocale } from '@/lib/route-helpers';

export async function GET() {
  return createRssResponse(rootLocale);
}
