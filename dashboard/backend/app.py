@app.route('/predict', methods=['POST'])
@jwt_required() # Protege o acesso
def predict():
    # 1. Receber dados do Streamlit (via request.json ou form-data)
    df_new = pd.read_excel(request.files['file']) 

    # 2. Executar o pipeline ML completo (loading dos .pkl e predict)
    df_results = preprocess_and_predict(df_new, models, scaler, FEATURES, CAT_COLS)

    # 3. SALVAR PERSISTÊNCIA (Novo requisito!)
    current_user_id = get_jwt_identity()
    for index, row in df_results.iterrows():
        # Lógica para salvar cada previsão de jogador no DB
        new_prediction = Prediction(
            user_id=current_user_id,
            jogador_id=row['Código de Acesso'],
            pred_t1=row['Previsão T1'],
            # ... salvar T2, T3 e o snapshot de dados
        )
        db.session.add(new_prediction)
    
    db.session.commit()
    
    # 4. Retornar os resultados finais para o Streamlit
    return jsonify(df_results.to_dict('records'))