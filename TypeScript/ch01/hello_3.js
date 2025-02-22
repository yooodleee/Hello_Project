/**
 * @typedef {Object} Article
 * @property {number} price
 * @property {number} vat
 * @property {string} string
 * @property {boolean=} sold
 */
/**
 * 이제 Article 형식으로 사용할 수 있다.
 * @param {[Article]} articles
 */
function totalAmount(articles) {
    return articles.reduce((total, article) => {
        return total + addVAT(article);
    }, 0);
}